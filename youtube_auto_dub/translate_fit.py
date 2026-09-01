"""Adapt a translated line so its spoken length fits the source time window.

Professional dubbing keeps the dubbed speech inside the same window the original
speaker used. When a literal translation is longer than that window, the dub is
either sped up (chipmunk) or cut (words lost) -> bad sync. This module shortens
the wording *while preserving meaning*, so the natural spoken length matches the
budget and the aligner needs little or no time-stretching.

Free, deterministic, no external model: contractions first, then removal of
low-information filler words, applied only as much as needed.
"""
from __future__ import annotations
import re

# Average English speaking rate (words per second) for conversational dubbing.
_WORDS_PER_SEC = 2.9

_CONTRACTIONS = {
    "do not": "don't", "does not": "doesn't", "did not": "didn't",
    "is not": "isn't", "are not": "aren't", "was not": "wasn't",
    "were not": "weren't", "cannot": "can't", "can not": "can't",
    "will not": "won't", "would not": "wouldn't", "should not": "shouldn't",
    "could not": "couldn't", "have not": "haven't", "has not": "hasn't",
    "had not": "hadn't", "must not": "mustn't", "it is": "it's",
    "that is": "that's", "there is": "there's", "he is": "he's",
    "she is": "she's", "what is": "what's", "who is": "who's",
    "they are": "they're", "we are": "we're", "you are": "you're",
    "i am": "I'm", "i will": "I'll", "you will": "you'll", "we will": "we'll",
    "they will": "they'll", "i have": "I've", "you have": "you've",
    "we have": "we've", "they have": "they've", "in order to": "to",
    "going to": "gonna", "want to": "wanna", "a lot of": "many",
    "because of": "due to", "at this point in time": "now",
    "in the event that": "if", "for the purpose of": "for",
}

# Low-information words that can be dropped without changing meaning.
_FILLERS = [
    "very", "really", "actually", "basically", "simply", "just", "quite",
    "in fact", "you know", "kind of", "sort of", "of course", "well",
    "so", "then", "even", "already", "still", "literally", "honestly",
]


def estimate_seconds(text: str, words_per_sec: float = _WORDS_PER_SEC) -> float:
    n = len([w for w in re.split(r"\s+", text.strip()) if w])
    return n / max(words_per_sec, 0.1)


def _apply_contractions(text: str) -> str:
    out = text
    for long, short in _CONTRACTIONS.items():
        out = re.sub(rf"\b{re.escape(long)}\b", short, out, flags=re.IGNORECASE)
    return out


def _drop_fillers(text: str, budget: float) -> str:
    out = text
    for f in _FILLERS:
        if estimate_seconds(out) <= budget:
            break
        out = re.sub(rf"\b{re.escape(f)}\b", "", out, flags=re.IGNORECASE)
    return out


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return text.strip()


def adapt_length(text: str, budget_seconds: float, target_lang: str = "en", tolerance: float = 1.12) -> str:
    """Shorten ``text`` so its spoken length fits ``budget_seconds`` (+tolerance).

    Only compresses English targets. Never expands, never invents words; if it
    cannot reach the budget with safe edits, it returns the best shorter form.
    """
    if not text or budget_seconds <= 0:
        return text
    if target_lang.lower() not in ("en", "en-us", "en-gb", "english"):
        return text
    limit = budget_seconds * tolerance
    if estimate_seconds(text) <= limit:
        return text
    step1 = _clean(_apply_contractions(text))
    if estimate_seconds(step1) <= limit:
        return step1
    step2 = _clean(_drop_fillers(step1, limit))
    return step2 or step1
