"""Quality pass for the next films: context-aware translation and smoother timing.

The first full film exposed three audible defects that were not bugs in any
single chunk but in how chunks meet:

* without Seed-VC nothing ever slowed short speech down, so a translated line
  that came out shorter than the original left a hole inside continuous
  narration (27 chunks with >0.5 s of dead air in the speech window);
* every chunk faded in and out over 15 ms although 124 of 156 cuts were made
  mid-sentence, so each seam got a small dent;
* one VoxCPM take came out 3.8x longer than its text and was crushed into the
  window instead of being generated again.

These tests pin the fixes and the optional LLM translator (enabled only when
``TRANSLATE_API_KEY`` is set) that sees the whole transcript, keeps names
consistent and respects each segment's spoken-time budget.
"""
from __future__ import annotations

import ast
import asyncio
import json
import os
import shutil
import subprocess
import wave
from pathlib import Path

import httpx
import numpy as np
import pytest

from youtube_auto_dub import llm_translate as lt

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/resumable_smart_dub.py"
WORKFLOW = ROOT / ".github/workflows/dub.yml"
VOXCPM = ROOT / "youtube_auto_dub/voxcpm_tts.py"


# ----------------------------------------------------------------- helpers
def _ffmpeg() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _write_wav(path: Path, seconds: float, rate: int = 24000, freq: float = 220.0, amplitude: float = 0.4) -> None:
    t = np.arange(int(round(seconds * rate))) / rate
    pcm = (amplitude * np.sin(2 * np.pi * freq * t) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())


def _wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def _extract(source: Path, names: set[str], namespace: dict) -> dict:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    constants = [
        n for n in tree.body
        if isinstance(n, ast.Assign) and all(isinstance(t, ast.Name) and t.id.isupper() for t in n.targets)
    ]
    exec(compile(ast.Module(body=constants + nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


def _timing_namespace(tmp_path: Path) -> dict:
    import soundfile as sf

    from youtube_auto_dub.content_validation import normalize_tokens
    from youtube_auto_dub.smart_chunks import atomic_write_json

    ffmpeg = _ffmpeg()
    if not ffmpeg:
        pytest.skip("ffmpeg is not available")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    link = bin_dir / "ffmpeg"
    if not link.exists():
        try:
            link.symlink_to(ffmpeg)
        except OSError:
            shutil.copy2(ffmpeg, link)
    os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
    namespace = {
        "Path": Path, "os": os, "shutil": shutil, "subprocess": subprocess, "sf": sf, "np": np,
        "SR_TTS": 24000, "atomic_write_json": atomic_write_json, "normalize_tokens": normalize_tokens,
        "CONTINUATION_CUTS": lt.CONTINUATION_CUTS,
    }
    _extract(SCRIPT, {
        "run", "atempo_filter", "fit_without_cutting", "match_duration_without_cutting",
        "match_duration_bounded", "plausible_tts_seconds", "boundary_fades", "build_chunk_audio",
    }, namespace)
    return namespace


class FakeAPI:
    """Scriptable OpenAI-compatible (or Anthropic) endpoint for httpx.MockTransport."""

    def __init__(self, replies, *, anthropic: bool = False):
        self.replies = list(replies)
        self.requests: list[dict] = []
        self.anthropic = anthropic

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        self.requests.append({"url": str(request.url), "headers": dict(request.headers), "body": body})
        reply = self.replies.pop(0)
        if isinstance(reply, httpx.Response):
            return reply
        if isinstance(reply, tuple):
            status, text = reply
            return httpx.Response(status, text=text)
        content = reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)
        if self.anthropic:
            return httpx.Response(200, json={"content": [{"type": "text", "text": content}], "usage": {"input_tokens": 10, "output_tokens": 5}})
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        })


def _translator(config: lt.LLMTranslateConfig, api: FakeAPI) -> lt.LLMTranslator:
    async def no_sleep(_seconds):
        return None

    return lt.LLMTranslator(config, transport=httpx.MockTransport(api.handler), sleep=no_sleep)


def _config(**overrides) -> lt.LLMTranslateConfig:
    base = dict(api_key="test-key", provider="openai", api_base="https://api.example.test/v1", model="test-model")
    base.update(overrides)
    return lt.LLMTranslateConfig(**base)


def _segments(*texts, seconds=6.0, continues=False):
    return [{"index": i, "text": text, "seconds": seconds, "continues": continues} for i, text in enumerate(texts)]


def _reply(mapping: dict[int, str]) -> dict:
    return {"segments": [{"id": key, "text": value} for key, value in mapping.items()]}


# ------------------------------------------------------------ configuration
def test_translator_is_off_without_an_api_key():
    assert lt.LLMTranslateConfig.from_env({}) is None
    assert lt.LLMTranslateConfig.from_env({"TRANSLATE_API_KEY": "   "}) is None


def test_provider_presets_and_overrides():
    config = lt.LLMTranslateConfig.from_env({"TRANSLATE_API_KEY": "k"})
    assert (config.provider, config.api_base, config.model) == ("openai", lt.PROVIDER_BASES["openai"], "gpt-4o-mini")
    config = lt.LLMTranslateConfig.from_env({"TRANSLATE_API_KEY": "k", "TRANSLATE_PROVIDER": "deepseek"})
    assert config.api_base == "https://api.deepseek.com/v1" and config.model == "deepseek-chat"
    config = lt.LLMTranslateConfig.from_env({
        "TRANSLATE_API_KEY": "k", "TRANSLATE_PROVIDER": "groq", "TRANSLATE_MODEL": "my-model",
        "TRANSLATE_STYLE": "documentary narration", "TRANSLATE_GLOSSARY": "نابليون=>Napoleon; النمسا => Austria",
        "TRANSLATE_WORDS_PER_SECOND": "2.5", "TRANSLATE_WINDOW": "12", "TRANSLATE_FALLBACK": "google",
    })
    assert config.model == "my-model" and config.words_per_second == 2.5 and config.window == 12
    assert config.glossary == {"نابليون": "Napoleon", "النمسا": "Austria"}
    assert config.fallback == "google" and config.style == "documentary narration"
    assert config.label == "llm:groq:my-model"


def test_anthropic_is_detected_from_the_base_url_and_custom_needs_a_base():
    config = lt.LLMTranslateConfig.from_env({"TRANSLATE_API_KEY": "k", "TRANSLATE_API_BASE": "https://api.anthropic.com/v1/"})
    assert config.provider == "anthropic" and config.api_base == "https://api.anthropic.com/v1"
    with pytest.raises(lt.TranslationConfigError):
        lt.LLMTranslateConfig.from_env({"TRANSLATE_API_KEY": "k", "TRANSLATE_PROVIDER": "custom"})
    with pytest.raises(lt.TranslationConfigError):
        lt.LLMTranslateConfig.from_env({"TRANSLATE_API_KEY": "k", "TRANSLATE_PROVIDER": "nope"})
    with pytest.raises(lt.TranslationConfigError):
        lt.LLMTranslateConfig.from_env({"TRANSLATE_API_KEY": "k", "TRANSLATE_FALLBACK": "silent"})
    custom = lt.LLMTranslateConfig.from_env({
        "TRANSLATE_API_KEY": "k", "TRANSLATE_API_BASE": "https://llm.internal/v1", "TRANSLATE_MODEL": "local",
    })
    assert custom.provider == "custom" and custom.model == "local"


def test_glossary_accepts_pairs_json_and_files(tmp_path):
    assert lt.parse_glossary("a=>b;c = d\ne=f") == {"a": "b", "c": "d", "e": "f"}
    assert lt.parse_glossary('{"x": "y"}') == {"x": "y"}
    path = tmp_path / "glossary.json"
    path.write_text(json.dumps({"تاليران": "Talleyrand"}), encoding="utf-8")
    assert lt.parse_glossary(str(path)) == {"تاليران": "Talleyrand"}
    assert lt.parse_glossary("") == {}


def test_reply_parsing_and_script_checks():
    assert lt.extract_json_object('```json\n{"segments": []}\n```') == {"segments": []}
    assert lt.extract_json_object('Sure! {"segments": [{"id": 1, "text": "x"}]} done')["segments"][0]["id"] == 1
    with pytest.raises(ValueError):
        lt.extract_json_object("no json here")
    assert lt.text_in_expected_script("Napoleon crossed the Alps.", "en", "ar")
    assert not lt.text_in_expected_script("Napoleon عبر the Alps", "en", "ar")
    assert not lt.text_in_expected_script("[موسيقى]", "en", "ar")
    assert not lt.text_in_expected_script("...", "en", "ar")
    assert lt.text_in_expected_script("عبر نابليون جبال الألب", "ar", "en")
    assert lt.word_budget(7.8, 2.7) == 21
    assert lt.word_budget(0.2, 2.7) == 2


# ------------------------------------------------------------- translation
def test_window_translation_carries_budget_context_and_checkpoints():
    api = FakeAPI([
        _reply({0: "And we begin.", 1: "Napoleon knew his father"}),
        _reply({2: "had served the monarchy."}),
    ])
    config = _config(window=2, context_segments=6)
    translator = _translator(config, api)
    segments = _segments("وهنا نبدأ", "عرف نابليون ان ابوه", "كان تبع الملكية", seconds=4.0, continues=True)
    segments[0]["continues"] = False
    saved = []

    results = asyncio.run(translator.translate_segments(
        segments, source_lang="ar", target_lang="en", on_window=lambda items: saved.append([i["index"] for i in items]),
    ))
    asyncio.run(translator.close())

    assert [r["text"] for r in results] == ["And we begin.", "Napoleon knew his father", "had served the monarchy."]
    assert all(r["engine"] == "llm:openai:test-model" for r in results)
    assert results[0]["max_words"] == lt.word_budget(4.0, 2.7) and results[0]["words"] == 3
    assert saved == [[0, 1], [2]]
    first, second = api.requests
    assert first["url"] == "https://api.example.test/v1/chat/completions"
    assert first["headers"]["authorization"] == "Bearer test-key"
    assert first["body"]["response_format"] == {"type": "json_object"}
    payload = json.loads(first["body"]["messages"][1]["content"])
    assert [s["id"] for s in payload["segments"]] == [0, 1]
    assert payload["segments"][1]["continues"] is True and payload["segments"][1]["max_words"] == 11
    assert "previous_segments_already_translated" not in payload
    system = first["body"]["messages"][0]["content"]
    assert "Arabic" in system and "English" in system and "max_words" in system
    context = json.loads(second["body"]["messages"][1]["content"])["previous_segments_already_translated"]
    assert [c["id"] for c in context] == [0, 1] and context[1]["translation"] == "Napoleon knew his father"


def test_prior_translations_seed_the_context_on_resume():
    api = FakeAPI([_reply({5: "then the guns fell silent."})])
    translator = _translator(_config(), api)
    prior = [{"index": 4, "text": "المدافع", "translation": "the guns opened fire,"}]
    results = asyncio.run(translator.translate_segments(
        [{"index": 5, "text": "وبعدين سكتت", "seconds": 3.0, "continues": False}],
        source_lang="ar", target_lang="en", prior=prior,
    ))
    asyncio.run(translator.close())
    assert results[0]["text"] == "then the guns fell silent."
    payload = json.loads(api.requests[0]["body"]["messages"][1]["content"])
    assert payload["previous_segments_already_translated"][0]["id"] == 4


def test_invalid_replies_are_rejected_and_asked_again():
    api = FakeAPI([
        _reply({0: "Only one of two"}),                        # missing id 1
        _reply({0: "Napoleon", 1: "خرج to war"}),               # source script leaked into the target
        _reply({0: "Napoleon", 1: "went to war"}),
    ])
    translator = _translator(_config(), api)
    results = asyncio.run(translator.translate_segments(_segments("نابليون", "خرج للحرب"), source_lang="ar", target_lang="en"))
    asyncio.run(translator.close())
    assert [r["text"] for r in results] == ["Napoleon", "went to war"]
    assert len(api.requests) == 3
    assert "rejected" in api.requests[1]["body"]["messages"][1]["content"]


def test_persistently_invalid_replies_fail_loudly():
    api = FakeAPI([_reply({7: "wrong id"})] * 3)
    translator = _translator(_config(), api)
    with pytest.raises(lt.TranslationAPIError, match="stayed invalid"):
        asyncio.run(translator.translate_segments(_segments("نص"), source_lang="ar", target_lang="en"))
    asyncio.run(translator.close())


def test_transient_errors_retry_and_permanent_errors_do_not():
    api = FakeAPI([(429, "slow down"), (503, "busy"), _reply({0: "Hello"})])
    translator = _translator(_config(max_retries=4), api)
    results = asyncio.run(translator.translate_segments(_segments("مرحبا"), source_lang="ar", target_lang="en"))
    asyncio.run(translator.close())
    assert results[0]["text"] == "Hello" and len(api.requests) == 3

    api = FakeAPI([(401, "bad key"), _reply({0: "never reached"})])
    translator = _translator(_config(), api)
    with pytest.raises(lt.TranslationAPIError) as info:
        asyncio.run(translator.translate_segments(_segments("مرحبا"), source_lang="ar", target_lang="en"))
    asyncio.run(translator.close())
    assert info.value.permanent and info.value.status == 401 and len(api.requests) == 1

    api = FakeAPI([(500, "down")] * 3)
    translator = _translator(_config(max_retries=2), api)
    with pytest.raises(lt.TranslationAPIError, match="after 3 attempts"):
        asyncio.run(translator.translate_segments(_segments("مرحبا"), source_lang="ar", target_lang="en"))
    asyncio.run(translator.close())


def test_json_mode_is_dropped_when_the_provider_rejects_it():
    api = FakeAPI([(400, '{"error": "response_format is not supported"}'), _reply({0: "Hello"})])
    translator = _translator(_config(), api)
    results = asyncio.run(translator.translate_segments(_segments("مرحبا"), source_lang="ar", target_lang="en"))
    asyncio.run(translator.close())
    assert results[0]["text"] == "Hello"
    assert "response_format" in api.requests[0]["body"] and "response_format" not in api.requests[1]["body"]


def test_over_budget_lines_get_one_compression_pass():
    long_line = " ".join(["word"] * 40)
    api = FakeAPI([
        _reply({0: long_line, 1: "Short enough."}),
        _reply({0: "Twelve words that now fit the budget of this line here."}),
    ])
    translator = _translator(_config(), api)
    results = asyncio.run(translator.translate_segments(
        _segments("جملة طويلة", "قصيرة", seconds=4.0), source_lang="ar", target_lang="en",
    ))
    asyncio.run(translator.close())
    assert results[0]["words"] == 11 and results[1]["text"] == "Short enough."
    compress = json.loads(api.requests[1]["body"]["messages"][1]["content"])
    assert [s["id"] for s in compress["segments"]] == [0] and compress["segments"][0]["current_words"] == 40

    # A failed or longer compression keeps the original rendering instead of failing the run.
    api = FakeAPI([_reply({0: long_line}), (500, "down"), (500, "down")])
    translator = _translator(_config(max_retries=1), api)
    results = asyncio.run(translator.translate_segments(_segments("جملة طويلة", seconds=4.0), source_lang="ar", target_lang="en"))
    asyncio.run(translator.close())
    assert results[0]["words"] == 40


def test_anthropic_messages_api_is_supported():
    api = FakeAPI([_reply({0: "Bonjour"})], anthropic=True)
    translator = _translator(_config(provider="anthropic", api_base="https://api.anthropic.com/v1"), api)
    results = asyncio.run(translator.translate_segments(_segments("hello"), source_lang="en", target_lang="fr"))
    asyncio.run(translator.close())
    assert results[0]["text"] == "Bonjour"
    request = api.requests[0]
    assert request["url"].endswith("/messages") and request["headers"]["x-api-key"] == "test-key"
    assert request["body"]["system"].startswith("You are a professional audiovisual translator")
    assert "response_format" not in request["body"]
    assert translator.usage == {"requests": 1, "prompt_tokens": 10, "completion_tokens": 5}


def test_preflight_reports_the_model_and_a_sample():
    api = FakeAPI([_reply({0: "The battle began at dawn, and nobody expected the outcome."})])
    translator = _translator(_config(), api)
    report = asyncio.run(translator.preflight(source_lang="ar", target_lang="en"))
    asyncio.run(translator.close())
    assert report["ok"] and report["model"] == "test-model" and report["sample_translation"].startswith("The battle")


# ----------------------------------------------------- pipeline integration
def test_pipeline_prefers_the_llm_translator_and_keeps_google_as_default():
    text = SCRIPT.read_text(encoding="utf-8")
    block = text.split("missing_translation = [chunk for chunk", 1)[1].split("# Pass 1:", 1)[0]
    assert "translate_with_llm(" in block
    assert "GoogleTranslator()" in block and 'translation_engine="google"' in block
    helper = text.split("async def translate_with_llm(", 1)[1].split("def parser(", 1)[0]
    assert "LLMTranslateConfig.from_env()" in helper
    assert "if config is None:\n        return pending" in helper
    assert '"continues": chunk.get("cut_reason") in CONTINUATION_CUTS' in helper
    assert "on_window=checkpoint" in helper and "mirror.upload_manifest(store)" in helper
    assert 'config.fallback == "google"' in helper
    assert 'translation_engine="label"' in helper
    # The translator is not part of the checkpoint config hash: resumed projects stay valid.
    config_block = text.split("    config = {", 1)[1].split("    }", 1)[0]
    assert "translat" not in config_block.lower()


def test_workflow_passes_translation_settings_and_preflights_them():
    import yaml

    text = WORKFLOW.read_text(encoding="utf-8")
    assert yaml.safe_load(text)
    preflight = text.split("- name: Preflight environment and credentials", 1)[1].split("- name: Upload preflight diagnostic", 1)[0]
    smart = text.split("- name: Run resumable speech-aware smart chunks", 1)[1].split("- name: Confirm complete resumable delivery", 1)[0]
    for section in (preflight, smart):
        assert "TRANSLATE_API_KEY: ${{ secrets.TRANSLATE_API_KEY }}" in section
        for name in ("TRANSLATE_PROVIDER", "TRANSLATE_API_BASE", "TRANSLATE_MODEL", "TRANSLATE_STYLE", "TRANSLATE_GLOSSARY", "TRANSLATE_FALLBACK"):
            assert f"{name}: ${{{{ vars.{name} }}}}" in section
    assert "python -m youtube_auto_dub.llm_translate --preflight" in preflight
    assert "output/translation-preflight.json" in preflight
    # No new workflow_dispatch input was needed (GitHub caps them at 25).
    section = text.split("    inputs:", 1)[1].split("\njobs:", 1)[0]
    import re
    assert len(re.findall(r"^      ([A-Za-z_][A-Za-z0-9_]*):\s*$", section, re.M)) <= 25


def test_preflight_cli_without_a_key_reports_the_default_engine(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("TRANSLATE_")}
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    report_path = tmp_path / "translation-preflight.json"
    proc = subprocess.run(
        [os.sys.executable, "-m", "youtube_auto_dub.llm_translate", "--preflight", "--report", str(report_path)],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] and report["engine"] == "google-unofficial"

    env["TRANSLATE_API_KEY"] = "k"
    env["TRANSLATE_PROVIDER"] = "nope"
    proc = subprocess.run(
        [os.sys.executable, "-m", "youtube_auto_dub.llm_translate", "--preflight"],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    assert proc.returncode == 2 and "unknown TRANSLATE_PROVIDER" in proc.stdout


# ------------------------------------------------------------------ timing
def test_short_speech_fills_its_window_but_never_slower_than_the_floor(tmp_path):
    ns = _timing_namespace(tmp_path)
    assert ns["SLOW_TEMPO_FLOOR"] == 0.85
    source = tmp_path / "take.wav"
    _write_wav(source, 1.0)

    # Mild shortfall: slowed to end with the original phrase.
    out, actual, fitted = ns["match_duration_bounded"](source, tmp_path / "fill-a.wav", 1.10)
    assert abs(actual - 1.0) < 0.002 and abs(fitted - 1.10) < 0.03

    # Large shortfall: stop at the floor (1.0 / 0.85 s) and leave the rest silent.
    out, actual, fitted = ns["match_duration_bounded"](source, tmp_path / "fill-b.wav", 3.0)
    assert abs(fitted - 1.0 / 0.85) < 0.03
    assert fitted < 1.25

    # Already matching: an untouched copy, no tempo pass.
    out, actual, fitted = ns["match_duration_bounded"](source, tmp_path / "fill-c.wav", 1.005)
    assert abs(fitted - 1.0) < 0.002 and abs(_wav_seconds(out) - 1.0) < 0.002

    # Too long: still accelerated to end with the phrase (Seed-VC sync behaviour) ...
    out, actual, fitted = ns["match_duration_bounded"](source, tmp_path / "fill-d.wav", 0.80)
    assert abs(fitted - 0.80) < 0.03
    # ... unless the caller forbids extra speed-up: the window fill only ever slows down.
    out, actual, fitted = ns["match_duration_bounded"](source, tmp_path / "fill-f.wav", 0.80, max_tempo=1.0)
    assert abs(fitted - 1.0) < 0.002 and abs(_wav_seconds(out) - 1.0) < 0.002

    with pytest.raises(RuntimeError):
        ns["match_duration_bounded"](source, tmp_path / "fill-e.wav", 0.05)


def test_window_fill_is_wired_into_the_voxcpm_only_path_and_seed_paths_are_bounded():
    text = SCRIPT.read_text(encoding="utf-8")
    render = text.split("current_stage = \"timing_fit\"", 1)[1].split("current_stage = \"content_validation\"", 1)[0]
    assert "if not seed_required:" in render
    assert 'directory / f"window-fill{variant}.wav"' in render
    assert 'timing_mode="window_fill_v1"' in render
    import re
    assert re.search(r"match_duration_bounded\(\s+fitted_voice, directory / f\"window-fill[^\n]*\n\s+min_tempo=SLOW_TEMPO_FLOOR, max_tempo=1\.0", render)
    assert re.search(r"match_duration_bounded\(\s+seed_voice, directory / f\"seedvc\.voice-only", render)
    assert not re.search(r"match_duration_without_cutting\(\s+seed_voice", render)
    retry = text.split("retry_trimmed = trim_generated(", 1)[1].split("current_stage = \"audio_mix\"", 1)[0]
    assert retry.count("match_duration_bounded(") == 2 and "match_duration_without_cutting(" not in retry
    # The window fill happens before the delivery fit, so nothing can exceed the chunk.
    assert render.index('window-fill{variant}.wav') < render.index('delivery{variant}.fitted.wav')
    # Seed-VC batch drift correction is untouched (converted vs. its own input).
    assert "synced, _actual, _fitted = match_duration_without_cutting(converted, synced, target_duration)" in text
    # Completed chunks are not forced to rebuild: no new mode gate was added.
    assert 'seed_mode_current = chunk.get("seed_vc_mode") == "voice_only_sync_v3"' in text
    assert text.count("timing_mode") == 1


def test_plausible_take_length_scales_with_the_text(tmp_path):
    ns = _timing_namespace(tmp_path)
    plausible = ns["plausible_tts_seconds"]
    assert abs(plausible("And we begin.") - (1.7 * 3 / 2.7 + 0.8)) < 1e-6
    twenty_two = " ".join(["word"] * 22)
    assert 13.0 < plausible(twenty_two) < 15.5   # the 29.5 s runaway take is far outside
    assert plausible("") > 0.8


def test_mid_sentence_seams_only_get_click_guards(tmp_path):
    ns = _timing_namespace(tmp_path)
    chunks = [
        {"cut_reason": "word_boundary"}, {"cut_reason": "natural_pause"},
        {"cut_reason": "word_boundary"}, {"cut_reason": "source_end"},
    ]
    fades = ns["boundary_fades"]
    assert fades(chunks, 0) == (0.015, 0.003)   # first chunk: normal in, continues out
    assert fades(chunks, 1) == (0.003, 0.015)   # continues in, pause out
    assert fades(chunks, 2) == (0.015, 0.003)
    assert fades(chunks, 3) == (0.003, 0.015)


def test_build_chunk_audio_applies_the_requested_fades(tmp_path):
    import soundfile as sf

    ns = _timing_namespace(tmp_path)
    voice = tmp_path / "voice.wav"
    _write_wav(voice, 0.5, amplitude=0.5)
    chunk = {"start": 10.0, "end": 11.0, "speech_start": 10.0, "speech_end": 10.5}

    ns["build_chunk_audio"](chunk, voice, None, tmp_path / "declick.wav", background_gain=0.0,
                            fade_in_seconds=0.003, fade_out_seconds=0.003)
    ns["build_chunk_audio"](chunk, voice, None, tmp_path / "edge.wav", background_gain=0.0)
    declick, _ = sf.read(str(tmp_path / "declick.wav"), dtype="float32")
    edge, _ = sf.read(str(tmp_path / "edge.wav"), dtype="float32")
    count = int(0.5 * 24000)

    def rms(values):
        return float(np.sqrt(np.mean(np.square(values.astype(np.float64)))))

    # 3 ms guard: by sample 100 (4 ms) the voice is at full level; the 15 ms fade is still ramping.
    head = slice(100, 300)
    assert np.max(np.abs(declick[head])) > 0.45
    assert rms(edge[head]) < 0.7 * rms(declick[head])
    # Tail: same shape mirrored.
    tail = slice(count - 300, count - 100)
    assert np.max(np.abs(declick[tail])) > 0.45
    assert rms(edge[tail]) < 0.7 * rms(declick[tail])
    report = json.loads((tmp_path / "mix-report.json").read_text(encoding="utf-8"))
    assert report["fade_in_seconds"] == 0.015 and report["fade_out_seconds"] == 0.015


def test_runaway_voxcpm_takes_are_regenerated_and_kept_for_audit(tmp_path, monkeypatch):
    from youtube_auto_dub import voxcpm_tts

    durations = iter([6.0, 5.0, 0.9])
    calls = []

    def fake_generate(text, dest, control, reference_audio):
        seconds = next(durations)
        calls.append(seconds)
        _write_wav(Path(dest), seconds)

    monkeypatch.setattr(voxcpm_tts, "_generate_local", fake_generate)
    monkeypatch.setattr(voxcpm_tts, "_BACKEND", "local")
    monkeypatch.setattr(voxcpm_tts, "_validate_generated_speech", lambda path: None)
    monkeypatch.setenv("YAD_VOXCPM_ATTEMPTS", "3")
    dest = tmp_path / "generated.wav"

    duration = asyncio.run(voxcpm_tts.speak_voxcpm("And we begin.", dest, max_seconds=2.7))
    assert abs(duration - 0.9) < 0.01 and abs(_wav_seconds(dest) - 0.9) < 0.01
    assert calls == [6.0, 5.0, 0.9]
    kept = sorted(p.name for p in tmp_path.glob("generated.long-take-*.wav"))
    assert kept == ["generated.long-take-1.wav", "generated.long-take-2.wav"]


def test_stubborn_long_lines_keep_the_shortest_take_instead_of_stalling(tmp_path, monkeypatch):
    from youtube_auto_dub import voxcpm_tts

    durations = iter([6.0, 4.0, 5.0, 0.5])
    calls = []

    def fake_generate(text, dest, control, reference_audio):
        seconds = next(durations)
        calls.append(seconds)
        _write_wav(Path(dest), seconds)

    monkeypatch.setattr(voxcpm_tts, "_generate_local", fake_generate)
    monkeypatch.setattr(voxcpm_tts, "_BACKEND", "local")
    monkeypatch.setattr(voxcpm_tts, "_validate_generated_speech", lambda path: None)
    monkeypatch.setenv("YAD_TTS_LONG_TAKE_ATTEMPTS", "3")
    dest = tmp_path / "generated.wav"

    duration = asyncio.run(voxcpm_tts.speak_voxcpm("Short line.", dest, max_seconds=2.0))
    assert calls == [6.0, 4.0, 5.0]
    assert abs(duration - 4.0) < 0.01 and abs(_wav_seconds(dest) - 4.0) < 0.01

    # Without a limit the first take is accepted unchanged.
    dest2 = tmp_path / "plain.wav"
    duration = asyncio.run(voxcpm_tts.speak_voxcpm("Short line.", dest2))
    assert abs(duration - 0.5) < 0.01


def test_transient_voxcpm_failures_still_retry_then_raise(tmp_path, monkeypatch):
    from youtube_auto_dub import voxcpm_tts

    calls = []

    def failing(text, dest, control, reference_audio):
        calls.append(1)
        raise RuntimeError("space busy")

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(voxcpm_tts, "_generate_local", failing)
    monkeypatch.setattr(voxcpm_tts, "_BACKEND", "local")
    monkeypatch.setattr(voxcpm_tts.asyncio, "sleep", no_sleep)
    monkeypatch.setenv("YAD_VOXCPM_ATTEMPTS", "3")
    with pytest.raises(RuntimeError, match="failed for en speech"):
        asyncio.run(voxcpm_tts.speak_voxcpm("x", tmp_path / "g.wav", max_seconds=3.0))
    assert len(calls) == 3


def test_synthesize_passes_the_plausible_length_to_voxcpm():
    text = SCRIPT.read_text(encoding="utf-8")
    block = text.split("async def synthesize(", 1)[1].split("async def translate_with_llm(", 1)[0]
    assert "max_seconds: float | None = None" in block
    assert "max_seconds = plausible_tts_seconds(text)" in block
    assert "max_seconds=max_seconds," in block
    assert "tts_duration_warning=bool(original_tts_duration > plausible_tts_seconds" in text
    vox = VOXCPM.read_text(encoding="utf-8")
    assert "def speak_voxcpm(text, dest, language=\"en\", control=\"\", reference_audio=None, max_seconds=None)" in vox
    assert "long-take-" in vox and ".unlink(" not in vox.split("duration = await asyncio.to_thread(_audio_seconds, dest)", 1)[1]
