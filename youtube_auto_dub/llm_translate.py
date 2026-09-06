"""Context-aware LLM translation for dubbing (optional).

The default translator (``googlev4.GoogleTranslator``) translates each smart
chunk in isolation. Chunks are cut every <=10 s, often mid-sentence, so the
translator never sees the sentence it is translating, cannot keep names
consistent, cannot repair speech-recognition slips, and has no idea how much
time the dubbed line may take. This module sends the *ordered transcript* to a
chat-completion model with a spoken-time budget per segment and asks for one
translation per segment.

Activation is explicit: the module is used only when ``TRANSLATE_API_KEY`` is
set (a repository secret in CI). Without it nothing changes.

Environment:
    TRANSLATE_API_KEY      required to enable the LLM translator
    TRANSLATE_PROVIDER     openai | deepseek | groq | openrouter | gemini |
                           mistral | together | xai | anthropic | agentrouter |
                           custom (default: openai, or anthropic when the base
                           URL points at api.anthropic.com). "gemini" (Google AI
                           Studio key, free tier) and "groq" (free tier) are
                           reachable from GitHub-hosted runners; agentrouter's
                           WAF is not (see PROJECT_NOTES). GitHub Models was
                           retired on 2026-07-30 and is deliberately absent.
    TRANSLATE_API_BASE     override the provider base URL (OpenAI-compatible
                           ``/chat/completions`` or Anthropic ``/messages``)
    TRANSLATE_MODEL        model name (provider default when omitted)
    TRANSLATE_EXTRA_HEADERS  extra HTTP headers some gateways demand, as a
                           JSON object or "Name: value; Name2: value2"
    TRANSLATE_MAX_TOKENS   completion budget per request (default 8192;
                           reasoning models return nothing when it is small)
    TRANSLATE_JSON_MODE    auto (default) | on | off — request JSON-object
                           replies; auto drops it after the first 400
    TRANSLATE_STYLE        free-text register hint, e.g. "documentary narration"
    TRANSLATE_GLOSSARY     "source=>target; source2=>target2" or a path to a
                           JSON object file with the same mapping
    TRANSLATE_WORDS_PER_SECOND  spoken pace used for the per-segment word
                           budget (default 2.7)
    TRANSLATE_WINDOW       segments per request (default 40)
    TRANSLATE_FALLBACK     "fail" (default) or "google": what to do when the
                           API keeps failing after retries
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx

PROVIDER_BASES = {
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "mistral": "https://api.mistral.ai/v1",
    "together": "https://api.together.xyz/v1",
    "xai": "https://api.x.ai/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "agentrouter": "https://agentrouter.org/v1",
}

PROVIDER_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "deepseek": "deepseek-chat",
    "groq": "llama-3.3-70b-versatile",
    "openrouter": "openai/gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
    "mistral": "mistral-large-latest",
    "together": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "xai": "grok-3-mini",
    "anthropic": "claude-3-5-haiku-latest",
    "agentrouter": "deepseek-v4-flash",
}

# Provider-specific request defaults. Gemini Flash models "think" before they
# answer and the thinking tokens count against max_tokens, so leave headroom.
PROVIDER_DEFAULT_LIMITS: dict[str, dict[str, int]] = {
    "gemini": {"max_tokens": 16384},
}

# Gateways with a WAF in front only admit requests that look like a known
# client; these defaults can be extended or overridden by TRANSLATE_EXTRA_HEADERS.
PROVIDER_DEFAULT_HEADERS: dict[str, dict[str, str]] = {
    "agentrouter": {"User-Agent": "codex_cli_rs/0.146.0", "originator": "codex_cli_rs"},
}

# Speech that continues straight into the next chunk (the planner had to cut
# inside a sentence). Translations of these pieces must read on naturally.
CONTINUATION_CUTS = {"word_boundary", "hard_limit_guard"}

LANGUAGE_NAMES = {
    "ar": "Arabic", "en": "English", "fr": "French", "de": "German", "es": "Spanish",
    "it": "Italian", "pt": "Portuguese", "tr": "Turkish", "ru": "Russian", "ja": "Japanese",
    "zh": "Chinese", "ko": "Korean", "hi": "Hindi", "ur": "Urdu", "fa": "Persian",
    "nl": "Dutch", "pl": "Polish", "sv": "Swedish", "id": "Indonesian", "vi": "Vietnamese",
}

_SCRIPTS = {
    "arabic": re.compile(r"[\u0600-\u06FF]"),
    "latin": re.compile(r"[A-Za-z\u00C0-\u024F]"),
    "cyrillic": re.compile(r"[\u0400-\u04FF]"),
    "cjk": re.compile(r"[\u3040-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF]"),
    "devanagari": re.compile(r"[\u0900-\u097F]"),
}
_LANGUAGE_SCRIPT = {
    "ar": "arabic", "fa": "arabic", "ur": "arabic",
    "en": "latin", "fr": "latin", "de": "latin", "es": "latin", "it": "latin", "pt": "latin",
    "tr": "latin", "nl": "latin", "pl": "latin", "sv": "latin", "id": "latin", "vi": "latin",
    "ru": "cyrillic", "ja": "cjk", "zh": "cjk", "ko": "cjk", "hi": "devanagari",
}

# 410 Gone is permanent by definition: a retired endpoint (GitHub Models, 2026-07)
# answers every request with it and a retry loop would spin forever.
PERMANENT_HTTP_ERRORS = {400, 401, 403, 404, 410, 422}


def _lang_code(value: str) -> str:
    return (value or "").strip().lower().split("-")[0]


def language_name(value: str) -> str:
    code = _lang_code(value)
    return LANGUAGE_NAMES.get(code, value or "the target language")


def count_words(text: str) -> int:
    return len([token for token in re.split(r"\s+", (text or "").strip()) if token])


def word_budget(seconds: float, words_per_second: float, minimum: int = 2) -> int:
    return max(minimum, int(round(max(0.0, float(seconds)) * float(words_per_second))))


def text_in_expected_script(text: str, target_lang: str, source_lang: str | None = None) -> bool:
    """True when ``text`` looks like ``target_lang`` and not like an untranslated source."""
    target_script = _LANGUAGE_SCRIPT.get(_lang_code(target_lang))
    source_script = _LANGUAGE_SCRIPT.get(_lang_code(source_lang or ""))
    stripped = re.sub(r"[\W\d_]+", "", text or "")
    if not stripped:
        return False
    if target_script and not _SCRIPTS[target_script].search(text):
        return False
    if source_script and source_script != target_script and _SCRIPTS[source_script].search(text):
        return False
    return True


def parse_glossary(value: str | None) -> dict[str, str]:
    raw = (value or "").strip()
    if not raw:
        return {}
    candidate = Path(raw)
    try:
        if candidate.suffix.lower() == ".json" and candidate.exists():
            data = json.loads(candidate.read_text(encoding="utf-8"))
            return {str(k).strip(): str(v).strip() for k, v in dict(data).items() if str(k).strip()}
        if raw.startswith("{"):
            data = json.loads(raw)
            return {str(k).strip(): str(v).strip() for k, v in dict(data).items() if str(k).strip()}
    except (OSError, ValueError, TypeError):
        return {}
    pairs: dict[str, str] = {}
    for item in re.split(r"[;\n]+", raw):
        if "=>" in item:
            source, target = item.split("=>", 1)
        elif "=" in item:
            source, target = item.split("=", 1)
        else:
            continue
        if source.strip() and target.strip():
            pairs[source.strip()] = target.strip()
    return pairs


def parse_headers(value: str | None) -> dict[str, str]:
    """Parse extra headers from a JSON object or ``Name: value; Name2: value2``."""
    raw = (value or "").strip()
    if not raw:
        return {}
    if raw.startswith("{"):
        try:
            data = json.loads(raw)
            return {str(k).strip(): str(v).strip() for k, v in dict(data).items() if str(k).strip()}
        except (ValueError, TypeError):
            return {}
    headers: dict[str, str] = {}
    for item in re.split(r"[;\n]+", raw):
        if ":" not in item:
            continue
        name, header_value = item.split(":", 1)
        if name.strip() and header_value.strip():
            headers[name.strip()] = header_value.strip()
    return headers


def extract_json_object(text: str) -> dict:
    """Return the first JSON object in a model reply (tolerates code fences)."""
    if not text:
        raise ValueError("empty model reply")
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except ValueError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("model reply contains no JSON object")
    data = json.loads(cleaned[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("model reply JSON is not an object")
    return data


def _retry_after_seconds(response: httpx.Response) -> float | None:
    raw = (response.headers.get("retry-after") or response.headers.get("x-ratelimit-reset-after") or "").strip()
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


class TranslationConfigError(RuntimeError):
    """The translator is enabled but cannot work with the given settings."""


class TranslationAPIError(RuntimeError):
    """The API rejected or failed the request."""

    def __init__(
        self, message: str, *, status: int | None = None, permanent: bool = False, retry_after: float | None = None,
    ):
        super().__init__(message)
        self.status = status
        self.permanent = permanent
        self.retry_after = retry_after


@dataclass
class LLMTranslateConfig:
    api_key: str
    provider: str = "openai"
    api_base: str = PROVIDER_BASES["openai"]
    model: str = PROVIDER_DEFAULT_MODELS["openai"]
    style: str = ""
    glossary: dict[str, str] = field(default_factory=dict)
    words_per_second: float = 2.7
    window: int = 40
    context_segments: int = 6
    timeout_seconds: float = 180.0
    max_retries: int = 5
    fallback: str = "fail"
    temperature: float = 0.2
    extra_headers: dict[str, str] = field(default_factory=dict)
    max_tokens: int = 8192
    json_mode: str = "auto"

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> "LLMTranslateConfig | None":
        env = os.environ if environ is None else environ
        base = (env.get("TRANSLATE_API_BASE") or "").strip().rstrip("/")
        provider = (env.get("TRANSLATE_PROVIDER") or "").strip().lower()
        if not provider:
            provider = "anthropic" if "anthropic.com" in base else ("custom" if base else "openai")
        if provider not in PROVIDER_BASES and provider != "custom":
            raise TranslationConfigError(
                f"unknown TRANSLATE_PROVIDER={provider!r}; expected one of {', '.join(sorted(PROVIDER_BASES))} or custom"
            )
        key = (env.get("TRANSLATE_API_KEY") or "").strip()
        if not key:
            return None
        if not base:
            if provider == "custom":
                raise TranslationConfigError("TRANSLATE_API_BASE is required when TRANSLATE_PROVIDER=custom")
            base = PROVIDER_BASES[provider]
        model = (env.get("TRANSLATE_MODEL") or "").strip() or PROVIDER_DEFAULT_MODELS.get(provider, "")
        if not model:
            raise TranslationConfigError("TRANSLATE_MODEL is required for a custom provider")
        fallback = (env.get("TRANSLATE_FALLBACK") or "fail").strip().lower()
        if fallback not in {"fail", "google"}:
            raise TranslationConfigError("TRANSLATE_FALLBACK must be 'fail' or 'google'")
        json_mode = (env.get("TRANSLATE_JSON_MODE") or "auto").strip().lower()
        if json_mode not in {"auto", "on", "off"}:
            raise TranslationConfigError("TRANSLATE_JSON_MODE must be auto, on or off")
        headers = dict(PROVIDER_DEFAULT_HEADERS.get(provider, {}))
        headers.update(parse_headers(env.get("TRANSLATE_EXTRA_HEADERS")))
        limits = PROVIDER_DEFAULT_LIMITS.get(provider, {})

        def _float(name: str, default: float) -> float:
            raw = (env.get(name) or "").strip()
            try:
                value = float(raw) if raw else default
            except ValueError as exc:
                raise TranslationConfigError(f"{name} must be a number") from exc
            if value <= 0:
                raise TranslationConfigError(f"{name} must be positive")
            return value

        return cls(
            api_key=key,
            provider=provider,
            api_base=base,
            model=model,
            style=(env.get("TRANSLATE_STYLE") or "").strip(),
            glossary=parse_glossary(env.get("TRANSLATE_GLOSSARY")),
            words_per_second=_float("TRANSLATE_WORDS_PER_SECOND", 2.7),
            window=max(1, int(_float("TRANSLATE_WINDOW", limits.get("window", 40)))),
            fallback=fallback,
            extra_headers=headers,
            max_tokens=max(256, int(_float("TRANSLATE_MAX_TOKENS", limits.get("max_tokens", 8192)))),
            json_mode=json_mode,
            max_retries=max(0, int(_float("TRANSLATE_MAX_RETRIES", 5))),
        )

    @property
    def label(self) -> str:
        return f"llm:{self.provider}:{self.model}"

    def public_summary(self) -> dict[str, Any]:
        return {
            "engine": "llm",
            "provider": self.provider,
            "model": self.model,
            "api_base": self.api_base,
            "words_per_second": self.words_per_second,
            "window": self.window,
            "style": self.style,
            "glossary_terms": len(self.glossary),
            "fallback": self.fallback,
            "extra_header_names": sorted(self.extra_headers),
            "max_tokens": self.max_tokens,
            "json_mode": self.json_mode,
        }


SYSTEM_PROMPT = """You are a professional audiovisual translator preparing a voice-over dub.
You receive an ordered list of transcript segments from {source_name} (automatic speech recognition, dialect and recognition errors included) and return one {target_name} translation per segment.

Rules:
1. Reply with JSON only, in the form {{"segments": [{{"id": <id>, "text": "<translation>"}}, ...]}} — exactly one entry per input id, same ids, same order, nothing else.
2. Each translation carries the meaning of ITS OWN segment. Do not move content into neighbouring segments except the few words grammar requires; never merge, drop, or add segments.
3. The speech is continuous and was cut into timed pieces. A segment with "continues": true ends mid-sentence and the next segment carries on; translate so consecutive pieces read as one natural sentence when spoken in order (a piece may end without a full stop).
4. Every segment has "max_words": the words that fit its time budget. Stay at or below it (shorter is fine). Prefer natural, concise spoken {target_name}; contractions are welcome. Faithfulness to meaning comes first, then brevity.
5. Use context to fix obvious speech-recognition slips and to keep names, places and terms consistent, spelled the standard {target_name} way. Keep numbers and quotations. Never leave words untranslated, never output {source_name} text, never add notes, brackets, or commentary.
6. Non-speech labels such as [music] or [applause] are returned as the equivalent {target_name} label."""


class LLMTranslator:
    """Chat-completion translator with per-segment time budgets and strict validation."""

    def __init__(
        self,
        config: LLMTranslateConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Any] | None = None,
    ):
        self.config = config
        self._client = httpx.AsyncClient(timeout=config.timeout_seconds, transport=transport)
        self._sleep = sleep or asyncio.sleep
        self._json_mode = config.provider != "anthropic" and config.json_mode != "off"
        self.usage: dict[str, int] = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0}

    async def close(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------ prompts
    def system_prompt(self, source_lang: str, target_lang: str) -> str:
        prompt = SYSTEM_PROMPT.format(
            source_name=language_name(source_lang), target_name=language_name(target_lang),
        )
        if self.config.style:
            prompt += f"\nRegister and style: {self.config.style}."
        if self.config.glossary:
            terms = "; ".join(f"{src} => {dst}" for src, dst in self.config.glossary.items())
            prompt += f"\nGlossary (always use these renderings): {terms}."
        return prompt

    def user_prompt(self, window: list[dict], context: list[dict]) -> str:
        payload: dict[str, Any] = {
            "segments": [
                {
                    "id": int(item["index"]),
                    "seconds": round(float(item.get("seconds") or 0.0), 2),
                    "max_words": int(item["max_words"]),
                    "continues": bool(item.get("continues")),
                    "text": str(item["text"]),
                }
                for item in window
            ]
        }
        if context:
            payload["previous_segments_already_translated"] = [
                {"id": int(item["index"]), "source": str(item["text"]), "translation": str(item["translation"])}
                for item in context
            ]
        return json.dumps(payload, ensure_ascii=False)

    # ---------------------------------------------------------------- transport
    async def _post(self, url: str, headers: dict[str, str], body: dict[str, Any]) -> httpx.Response:
        try:
            return await self._client.post(url, headers=headers, json=body)
        except httpx.HTTPError as exc:
            raise TranslationAPIError(f"translation API request failed: {exc}") from exc

    async def complete(self, system: str, user: str) -> str:
        """Send one chat completion and return the text reply (with retries)."""
        config = self.config
        last_error: Exception | None = None
        for attempt in range(config.max_retries + 1):
            try:
                if config.provider == "anthropic":
                    headers = {"x-api-key": config.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
                    headers.update(config.extra_headers)
                    response = await self._post(
                        f"{config.api_base}/messages",
                        headers,
                        {
                            "model": config.model, "max_tokens": config.max_tokens, "temperature": config.temperature,
                            "system": system, "messages": [{"role": "user", "content": user}],
                        },
                    )
                else:
                    body: dict[str, Any] = {
                        "model": config.model,
                        "temperature": config.temperature,
                        "max_tokens": config.max_tokens,
                        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    }
                    if self._json_mode:
                        body["response_format"] = {"type": "json_object"}
                    headers = {"Authorization": f"Bearer {config.api_key}", "content-type": "application/json"}
                    headers.update(config.extra_headers)
                    response = await self._post(f"{config.api_base}/chat/completions", headers, body)
                self.usage["requests"] += 1
                if response.status_code == 400 and self._json_mode and config.json_mode == "auto":
                    # Gateways differ in what they accept; JSON mode is only a
                    # nicety, the prompt itself demands JSON. Drop it once and retry.
                    self._json_mode = False
                    continue
                if response.status_code in PERMANENT_HTTP_ERRORS:
                    raise TranslationAPIError(
                        f"translation API rejected the request ({response.status_code}): {response.text[:300]}",
                        status=response.status_code, permanent=True,
                    )
                if response.status_code >= 400:
                    raise TranslationAPIError(
                        f"translation API error {response.status_code}: {response.text[:300]}",
                        status=response.status_code, retry_after=_retry_after_seconds(response),
                    )
                try:
                    data = response.json()
                except ValueError as exc:
                    raise TranslationAPIError(f"translation API returned non-JSON: {response.text[:300]}") from exc
                return self._extract_text(data)
            except TranslationAPIError as exc:
                if exc.permanent:
                    raise
                last_error = exc
                if attempt >= config.max_retries:
                    break
                # Rate limits (429) usually say how long to wait; honour that,
                # otherwise back off exponentially: 3, 6, 12, 24, 48 s.
                delay = min(3.0 * (2.0 ** attempt), 60.0)
                if exc.retry_after is not None:
                    delay = min(max(delay, exc.retry_after), 90.0)
                await self._sleep(delay)
        raise TranslationAPIError(f"translation API failed after {config.max_retries + 1} attempts: {last_error}")

    def _extract_text(self, data: dict) -> str:
        usage = data.get("usage") or {}
        self.usage["prompt_tokens"] += int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        self.usage["completion_tokens"] += int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        if self.config.provider == "anthropic":
            parts = data.get("content") or []
            text = "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict))
        else:
            choices = data.get("choices") or []
            if not choices:
                raise TranslationAPIError(f"translation API returned no choices: {json.dumps(data)[:300]}")
            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, list):
                text = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
            else:
                text = str(content or "")
        if not text.strip():
            raise TranslationAPIError("translation API returned an empty reply")
        return text

    # --------------------------------------------------------------- translate
    @staticmethod
    def iter_windows(segments: list[dict], size: int) -> list[list[dict]]:
        size = max(1, int(size))
        return [segments[offset:offset + size] for offset in range(0, len(segments), size)]

    def prepare(self, segments: list[dict]) -> list[dict]:
        prepared = []
        for item in segments:
            entry = dict(item)
            entry["max_words"] = word_budget(float(entry.get("seconds") or 0.0), self.config.words_per_second)
            prepared.append(entry)
        return prepared

    def _validate_reply(self, window: list[dict], reply: str, *, source_lang: str, target_lang: str) -> dict[int, str]:
        data = extract_json_object(reply)
        items = data.get("segments")
        if not isinstance(items, list):
            raise ValueError("reply JSON has no 'segments' list")
        expected = [int(item["index"]) for item in window]
        got: dict[int, str] = {}
        for item in items:
            if not isinstance(item, dict) or "id" not in item:
                raise ValueError("reply segment without id")
            try:
                identifier = int(item["id"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"reply segment id is not an integer: {item.get('id')!r}") from exc
            got[identifier] = str(item.get("text") or "").strip()
        missing = [identifier for identifier in expected if identifier not in got]
        extra = [identifier for identifier in got if identifier not in expected]
        if missing or extra:
            raise ValueError(f"reply ids mismatch: missing={missing[:10]} extra={extra[:10]}")
        for identifier in expected:
            text = got[identifier]
            if not text:
                raise ValueError(f"empty translation for segment {identifier}")
            if not text_in_expected_script(text, target_lang, source_lang):
                raise ValueError(f"segment {identifier} is not in {language_name(target_lang)}: {text[:60]!r}")
        return got

    async def translate_window(
        self, window: list[dict], context: list[dict], *, source_lang: str, target_lang: str,
    ) -> list[dict]:
        """Translate one window of prepared segments; returns dicts with index/text/words/max_words."""
        system = self.system_prompt(source_lang, target_lang)
        user = self.user_prompt(window, context)
        last_error: Exception | None = None
        translations: dict[int, str] | None = None
        for attempt in range(3):
            reply = await self.complete(system, user)
            try:
                translations = self._validate_reply(window, reply, source_lang=source_lang, target_lang=target_lang)
                break
            except ValueError as exc:
                last_error = exc
                user = self.user_prompt(window, context) + (
                    f"\n\nYour previous reply was rejected: {exc}. Reply again with valid JSON covering every id exactly once."
                )
        if translations is None:
            raise TranslationAPIError(f"model replies stayed invalid: {last_error}")

        over = [item for item in window if count_words(translations[int(item["index"])]) > int(item["max_words"]) * 1.35 + 1]
        if over:
            translations = await self._compress(over, translations, source_lang=source_lang, target_lang=target_lang)

        results = []
        for item in window:
            index = int(item["index"])
            text = translations[index]
            results.append({
                "index": index,
                "text": text,
                "words": count_words(text),
                "max_words": int(item["max_words"]),
                "engine": self.config.label,
            })
        return results

    async def _compress(
        self, over: list[dict], translations: dict[int, str], *, source_lang: str, target_lang: str,
    ) -> dict[int, str]:
        """Ask for shorter renderings of the segments that overshoot their budget."""
        target_name = language_name(target_lang)
        system = (
            f"You shorten {target_name} dubbing lines so they fit a time budget. Keep the full meaning of the source, "
            f"drop filler and redundancy, use contractions, keep names and numbers. Reply with JSON only: "
            f'{{"segments": [{{"id": <id>, "text": "<shorter {target_name} line>"}}]}}.'
        )
        payload = {
            "segments": [
                {
                    "id": int(item["index"]),
                    "max_words": int(item["max_words"]),
                    "source": str(item["text"]),
                    "current_translation": translations[int(item["index"])],
                    "current_words": count_words(translations[int(item["index"])]),
                }
                for item in over
            ]
        }
        try:
            reply = await self.complete(system, json.dumps(payload, ensure_ascii=False))
            shorter = self._validate_reply(over, reply, source_lang=source_lang, target_lang=target_lang)
        except (TranslationAPIError, ValueError):
            return translations
        updated = dict(translations)
        for item in over:
            index = int(item["index"])
            candidate = shorter.get(index, "")
            if candidate and count_words(candidate) < count_words(updated[index]):
                updated[index] = candidate
        return updated

    async def translate_segments(
        self, segments: list[dict], *, source_lang: str, target_lang: str,
        prior: list[dict] | None = None, on_window: Callable[[list[dict]], Any] | None = None,
    ) -> list[dict]:
        """Translate all ``segments`` in windows, carrying translated context forward.

        ``prior`` are already-translated segments (index/text/translation) that
        precede the pending ones; they seed the context on resumed runs.
        ``on_window`` is awaited/called after every window so callers can
        checkpoint partial progress.
        """
        prepared = self.prepare(segments)
        context: list[dict] = list(prior or [])[-self.config.context_segments:]
        results: list[dict] = []
        for window in self.iter_windows(prepared, self.config.window):
            translated = await self.translate_window(window, context, source_lang=source_lang, target_lang=target_lang)
            results.extend(translated)
            by_index = {item["index"]: item for item in translated}
            context = (context + [
                {"index": item["index"], "text": item["text"], "translation": by_index[int(item["index"])]["text"]}
                for item in window
            ])[-self.config.context_segments:]
            if on_window is not None:
                outcome = on_window(translated)
                if asyncio.iscoroutine(outcome):
                    await outcome
        return results

    async def preflight(self, *, source_lang: str, target_lang: str, sample: str | None = None) -> dict[str, Any]:
        """Translate one sentence to prove the credentials and model work."""
        text = sample or {
            "ar": "بدأت المعركة عند الفجر، ولم يتوقع أحد النتيجة.",
            "en": "The battle began at dawn, and nobody expected the outcome.",
        }.get(_lang_code(source_lang), "The battle began at dawn, and nobody expected the outcome.")
        started = time.monotonic()
        results = await self.translate_window(
            self.prepare([{"index": 0, "text": text, "seconds": 4.0, "continues": False}]),
            [], source_lang=source_lang, target_lang=target_lang,
        )
        return {
            "ok": True,
            "provider": self.config.provider,
            "model": self.config.model,
            "latency_seconds": round(time.monotonic() - started, 2),
            "sample_source": text,
            "sample_translation": results[0]["text"],
            "requests": self.usage["requests"],
        }


def _cli() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="LLM translation preflight")
    parser.add_argument("--preflight", action="store_true", help="translate one sentence with the configured API")
    parser.add_argument("--source-lang", default="ar")
    parser.add_argument("--target-lang", default="en")
    parser.add_argument("--report", type=Path, help="write the JSON report here")
    args = parser.parse_args()
    try:
        config = LLMTranslateConfig.from_env()
    except TranslationConfigError as exc:
        print(json.dumps({"ok": False, "engine": "llm", "error": str(exc)}, ensure_ascii=False))
        return 2
    if config is None:
        provider = (os.environ.get("TRANSLATE_PROVIDER") or "").strip().lower()
        note = "TRANSLATE_API_KEY not set; using the built-in Google Translate client"
        if provider:
            note = f"TRANSLATE_PROVIDER={provider} but TRANSLATE_API_KEY is not set; using the built-in Google Translate client"
        report = {"ok": True, "engine": "google-unofficial", "note": note}
        print(json.dumps(report, ensure_ascii=False))
        if args.report:
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0
    report: dict[str, Any] = {"engine": "llm", **config.public_summary()}
    if args.preflight:
        async def _run() -> dict[str, Any]:
            translator = LLMTranslator(config)
            try:
                return await translator.preflight(source_lang=args.source_lang, target_lang=args.target_lang)
            finally:
                await translator.close()

        try:
            report.update(asyncio.run(_run()))
        except Exception as exc:  # noqa: BLE001 - surfaced verbatim to the operator
            report.update({"ok": False, "error": str(exc)})
    code = preflight_exit_code(report, config)
    if code == 0 and not report.get("ok", True):
        report["note"] = "translation API unreachable; TRANSLATE_FALLBACK=google will translate with the built-in client"
        print(json.dumps({"warning": report["note"]}, ensure_ascii=False), file=sys.stderr)
    print(json.dumps(report, ensure_ascii=False))
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return code


def preflight_exit_code(report: dict[str, Any], config: LLMTranslateConfig | None) -> int:
    """0 when the run may proceed: the API works, or the operator opted into the Google fallback."""
    if report.get("ok", True):
        return 0
    if config is not None and config.fallback == "google":
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(_cli())
