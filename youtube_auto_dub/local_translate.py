"""Offline English → Arabic translator used when Google Translate is blocked."""

from __future__ import annotations

import re
from typing import List

PHRASES = [
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
    "welcome": "أهلاً",
    "hello": "مرحباً",
    "hi": "أهلاً",
    "everyone": "الجميع",
    "thanks": "شكراً",
    "thank": "شكر",
    "you": "أنت",
    "your": "ـك",
    "we": "نحن",
    "i": "أنا",
    "they": "هم",
    "this": "هذا",
    "that": "ذلك",
    "these": "هذه",
    "those": "تلك",
    "is": "يكون",
    "are": "تكون",
    "was": "كان",
    "were": "كانوا",
    "be": "يكون",
    "to": "إلى",
    "of": "من",
    "and": "و",
    "or": "أو",
    "in": "في",
    "on": "على",
    "for": "من أجل",
    "with": "مع",
    "from": "من",
    "by": "بواسطة",
    "as": "كـ",
    "at": "في",
    "an": "",
    "a": "",
    "the": "",
    "short": "قصير",
    "film": "فيلم",
    "video": "فيديو",
    "movie": "فيلم",
    "shows": "يعرض",
    "show": "عرض",
    "automatic": "آلي",
    "auto": "آلي",
    "dubbing": "دبلجة",
    "dub": "دبلجة",
    "translation": "ترجمة",
    "translate": "ترجم",
    "meaning": "المعنى",
    "speech": "الكلام",
    "voice": "صوت",
    "new": "جديد",
    "arabic": "العربية",
    "english": "الإنجليزية",
    "first": "أولاً",
    "then": "ثم",
    "finally": "أخيراً",
    "generate": "نولّد",
    "sync": "نزامن",
    "picture": "الصورة",
    "image": "صورة",
    "today": "اليوم",
    "now": "الآن",
    "here": "هنا",
    "how": "كيف",
    "what": "ماذا",
    "why": "لماذا",
    "when": "متى",
    "where": "أين",
    "who": "من",
    "can": "يمكن",
    "will": "سوف",
    "not": "لا",
    "no": "لا",
    "yes": "نعم",
    "please": "من فضلك",
    "good": "جيد",
    "great": "رائع",
    "best": "الأفضل",
    "free": "مجاني",
    "open": "افتح",
    "source": "مصدر",
    "channel": "قناة",
    "watch": "شاهد",
    "watching": "المشاهدة",
    "like": "أعجب",
    "share": "شارك",
    "subscribe": "اشترك",
    "comment": "علّق",
    "next": "التالي",
    "before": "قبل",
    "after": "بعد",
    "about": "حول",
    "into": "إلى",
    "it": "إنه",
    "its": "ـه",
    "our": "ـنا",
    "my": "ـي",
    "me": "أنا",
    "us": "نحن",
    "them": "هم",
    "his": "ـه",
    "her": "ـها",
    "she": "هي",
    "he": "هو",
    "do": "افعل",
    "does": "يفعل",
    "did": "فعل",
    "have": "لديه",
    "has": "لديه",
    "had": "كان لديه",
    "make": "اصنع",
    "made": "صنع",
    "use": "استخدم",
    "using": "باستخدام",
    "work": "عمل",
    "works": "يعمل",
    "working": "يعمل",
    "ready": "جاهز",
    "start": "ابدأ",
    "started": "بدأ",
    "test": "اختبار",
    "studio": "ستوديو",
    "prostudio": "برو ستوديو",
    "pro": "برو",
    "step": "خطوة",
    "steps": "خطوات",
    "download": "تنزيل",
    "upload": "رفع",
    "file": "ملف",
    "audio": "صوت",
    "music": "موسيقى",
    "background": "خلفية",
    "time": "وقت",
    "second": "ثانية",
    "seconds": "ثوانٍ",
    "minute": "دقيقة",
    "minutes": "دقائق",
    "people": "الناس",
    "world": "العالم",
    "life": "الحياة",
    "day": "يوم",
    "year": "سنة",
    "one": "واحد",
    "two": "اثنان",
    "three": "ثلاثة",
    "four": "أربعة",
    "five": "خمسة",
    "six": "ستة",
    "seven": "سبعة",
    "eight": "ثمانية",
    "nine": "تسعة",
    "ten": "عشرة",
}


def _apply_phrases(text: str) -> str:
    out = text
    for src, dst in sorted(PHRASES, key=lambda x: len(x[0]), reverse=True):
        out = re.sub(re.escape(src), dst, out, flags=re.IGNORECASE)
    return out


def _word(token: str) -> str:
    raw = token.strip()
    if not raw:
        return ""
    punct = ""
    while raw and raw[-1] in ".,!?;:":
        punct = raw[-1] + punct
        raw = raw[:-1]
    key = re.sub(r"[^A-Za-z']", "", raw).lower()
    if not key:
        return token
    if key in WORDS:
        return (WORDS[key] + punct).strip()
    return token


def translate_offline(text: str, source: str = "en", target: str = "ar") -> str:
    if not text or not text.strip():
        return text
    if source != "auto" and source == target:
        return text
    if target != "ar":
        return text
    lowered = text.strip()
    phrased = _apply_phrases(lowered)
    # If the whole sentence was replaced, keep it.
    if re.search(r"[\u0600-\u06FF]", phrased) and not re.search(r"[A-Za-z]", phrased):
        return phrased
    parts = re.split(r"(\s+)", phrased)
    built = []
    for part in parts:
        if part.isspace() or not part:
            built.append(part)
        elif re.search(r"[\u0600-\u06FF]", part):
            built.append(part)
        else:
            built.append(_word(part))
    result = "".join(built)
    result = re.sub(r"\s{2,}", " ", result).strip()
    result = re.sub(r"\s+([،,.!?])", r"\1", result)
    return result or text


def translate_batch_offline(texts: List[str], source: str = "en", target: str = "ar") -> List[str]:
    return [translate_offline(t, source=source, target=target) for t in texts]
