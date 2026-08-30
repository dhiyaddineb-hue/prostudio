"""Conservative, deterministic prosody hints for dubbed speech.

This deliberately returns neutral when evidence is weak. It is a cue for the
TTS model, not a claim that emotion recognition is perfect.
"""
from __future__ import annotations

import re

_INTENSE = re.compile(r"\b(urgent|stop|danger|never|impossible|غاضب|خطر|توقف|مستحيل)\b", re.I)
_SAD = re.compile(r"\b(sad|sorry|lost|miss|cry|حزين|آسف|فقد|اشتاق|بكاء)\b", re.I)
_HAPPY = re.compile(r"\b(great|amazing|love|happy|wonderful|رائع|سعيد|أحب|مذهل)\b", re.I)
_QUESTION = re.compile(r"[؟?]")
_EXCITED = re.compile(r"[!！]{1,}|\b(wow|really|yes)\b", re.I)


def infer_emotion(text: str) -> str:
    """Return a low-risk style label suitable for a TTS control prompt."""
    value = (text or "").strip()
    if not value:
        return "neutral, natural conversational delivery"
    if _INTENSE.search(value):
        return "urgent and emphatic, but still intelligible"
    if _SAD.search(value):
        return "subdued and reflective, with gentle pacing"
    if _HAPPY.search(value) or _EXCITED.search(value):
        return "warm and lightly enthusiastic, with natural energy"
    if _QUESTION.search(value):
        return "curious and conversational, with a clear question contour"
    return "neutral, natural conversational delivery"
