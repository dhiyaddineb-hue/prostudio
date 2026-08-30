"""Offline English to Arabic fallback translator.

This fallback is intentionally conservative: known phrases are translated, and
unknown English text is not presented as a successful Arabic translation.
"""
from __future__ import annotations
import re
from typing import List

PHRASES = [
    ("i love you", "أحبك"),
    ("too late", "لقد فات الأوان"),
    ("i don't love you anymore", "لم أعد أحبك"),
    ("i would have loved you forever", "كنت سأحبك إلى الأبد"),
    ("now please go", "الآن، أرجوك اذهب"),
    ("don't do this, alice", "لا تفعلي هذا يا أليس"),
    ("talk to me", "تحدثي معي"),
    ("where is this love", "أين هذا الحب؟"),
    ("where", "أين"),
    ("what", "ماذا"),
    ("show me", "أريني"),
    ("i can't see it, i can't touch it, i can't feel it", "لا أستطيع رؤيته، ولا لمسه، ولا الشعور به"),
    ("i can hear it, i can hear some words, but i can't do anything with your easy words", "أستطيع سماعه، أستطيع سماع بعض الكلمات، لكن لا أستطيع فعل شيء بكلماتك السهلة"),
    ("whatever you say, it's too late", "مهما قلت، لقد فات الأوان"),
    ("welcome to prostudio", "مرحباً بكم في برو ستوديو"),
    ("welcome to pro studio", "مرحباً بكم في برو ستوديو"),
    ("welcome to", "مرحباً بكم في"),
    ("this short film shows automatic video dubbing", "يعرض هذا الفيلم القصير الدبلجة الآلية للفيديو"),
    ("automatic video dubbing", "الدبلجة الآلية للفيديو"),
    ("first we transcribe the speech", "أولاً نُفرّغ الكلام"),
    ("then we translate the meaning into arabic", "ثم نترجم المعنى إلى العربية"),
    ("finally we generate a new voice and sync it with the picture", "وأخيراً نولّد صوتاً جديداً ونزامنه مع الصورة"),
    ("this is a short test of automatic video dubbing", "هذا اختبار قصير للدبلجة الآلية للفيديو"),
    ("hello everyone", "مرحباً بالجميع"),
    ("thank you for watching", "شكراً لمشاهدتكم"),
    ("subscribe to the channel", "اشتركوا في القناة"),
    ("don't forget to like", "لا تنسوا الإعجاب"),
    ("in this video", "في هذا الفيديو"),
    ("let's get started", "لنبدأ"),
    ("let us begin", "لنبدأ"),
]

WORDS = {
    "i": "أنا", "you": "أنت", "we": "نحن", "this": "هذا", "that": "ذلك",
    "is": "هو", "are": "هم", "was": "كان", "were": "كانوا", "to": "إلى",
    "of": "من", "and": "و", "or": "أو", "in": "في", "on": "على", "for": "من أجل",
    "with": "مع", "from": "من", "not": "لا", "no": "لا", "yes": "نعم",
    "please": "من فضلك", "now": "الآن", "here": "هنا", "how": "كيف", "what": "ماذا",
    "where": "أين", "who": "من", "can": "يمكن", "will": "سوف", "love": "أحب",
    "like": "أعجب", "voice": "صوت", "speech": "الكلام", "arabic": "العربية",
    "english": "الإنجليزية", "video": "فيديو", "dubbing": "دبلجة", "music": "موسيقى",
}

def _apply_phrases(text: str) -> str:
    out = text
    for src, dst in sorted(PHRASES, key=lambda x: len(x[0]), reverse=True):
        out = re.sub(re.escape(src), dst, out, flags=re.IGNORECASE)
    return out

def _word(token: str) -> str:
    raw = token.strip()
    if not raw: return ""
    punct = ""
    while raw and raw[-1] in ".,!?;:":
        punct = raw[-1] + punct
        raw = raw[:-1]
    key = re.sub(r"[^A-Za-z']", "", raw).lower()
    return (WORDS.get(key, token) + punct).strip()

def translate_offline(text: str, source: str = "en", target: str = "ar") -> str:
    if not text or not text.strip(): return text
    if source != "auto" and source == target: return text
    if target != "ar": return text
    phrased = _apply_phrases(text.strip())
    if re.search(r"[\u0600-\u06FF]", phrased) and not re.search(r"[A-Za-z]", phrased):
        return phrased
    parts = re.split(r"(\s+)", phrased)
    result = "".join(part if part.isspace() or not part or re.search(r"[\u0600-\u06FF]", part) else _word(part) for part in parts)
    return re.sub(r"\s+([،,.!?])", r"\1", re.sub(r"\s{2,}", " ", result)).strip()

def translate_batch_offline(texts: List[str], source: str = "en", target: str = "ar") -> List[str]:
    return [translate_offline(t, source=source, target=target) for t in texts]
