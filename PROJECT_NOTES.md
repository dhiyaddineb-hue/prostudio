# مذكرة المشروع — ProStudio
## الفرع: arena/01a03969-prostudio

---

## 📁 هيكل المشروع الكامل

### 1. النواة البرمجية (`youtube_auto_dub/`)
| الملف | الوظيفة |
|---|---|
| `__init__.py` | إصدار المشروع |
| `__main__.py` | نقطة تشغيل `python -m youtube_auto_dub` |
| `cli.py` | واجهة سطر الأوامر — argparse + تشغيل `core.run()` |
| `core.py` | **الخط الأنبوبي الرئيسي**: تنزيل → تفريغ → ترجمة → TTS → مزج → تصدير |
| `models.py` | **كل الإعدادات والثوابت**: مسارات، عينات الصوت، إعدادات Whisper/VAD/TTS/FFmpeg، شخصيات صوتية |
| `speech.py` | تفريغ الصوت بـ Faster-Whisper + VAD |
| `vosk_asr.py` | تفريغ بـ Vosk (أوفلاين، إنجليزي) — يحمل الموديل من GitHub |
| `offline_asr.py` | تفريغ بـ PocketSphinx (أوفلاين، إنجليزي، دقيقته ضعيفة) |
| `voice.py` | **محرك TTS متعدد**: Edge-TTS (أساسي) + Qwen3-TTS (Chatterbox) + استنساخ صوتي |
| `local_tts.py` | بديل أوفلاين: eSpeak NG (روبوتي) |
| `studio_tts.py` | بنك أصوات جاهزة في `samples/voices/` |
| `clone_tts.py` | استنساخ عصبي بـ F5-TTS (إنجليزي/صيني فقط) |
| `xtts_clone.py` | **XTTS-v2** — استنساخ يدعم العربية! يحتاج تحميل من HuggingFace |
| `voice_profile.py` | قياس بصمة صوتية (F0 + Formants) بـ Praat + تحويل أكوستيكي |
| `voice_design.py` | **مصمم الأصوات**: 6 عوامل تحكم (pitch, rate, body, warmth, clarity, air) + وصف عربي + presets |
| `diarize.py` | تحديد المتحدثين (MFCC + LDA) |
| `stem_split.py` | فصل الحوار عن الموسيقى (معالجة إشارة DSP — بلا موديل) |
| `audio.py` | معالجة الصوت: تقسيم، مزامنة tempo، تطبيع loudness، مزج خلفية، تصدير فيديو |
| `googlev4.py` | ترجمة Google |
| `local_translate.py` | ترجمة محلية (أوفلاين) |
| `arabic_text.py` | تجهيز النص العربي (تشكيل) |
| `align_text.py` | محاذاة النص مع الصوت |
| `subs.py` | قراءة/كتابة SRT |
| `youtube.py` | تنزيل من يوتيوب بـ yt-dlp |
| `ffmpeg_bin.py` | إيجاد FFmpeg (imageio-ffmpeg مضمّن) |
| `runtime.py` | كشف القدرات (GPU, Whisper, Edge-TTS, eSpeak) |
| `project_dirs.py` | إدارة مشاريع الدبلجة |
| `pipeline_args.py` | بناء معاملات الخط الأنبوبي |
| `ui.py` | واجهة طرفية (Rich) |
| `language_map.json` | خريطة اللغات + الأصوات المتاحة لكل لغة |

### 2. واجهة الويب (`web/`)
| الملف | الوظيفة |
|---|---|
| `server.py` | تشغيل uvicorn على المنفذ 8080 |
| `app.py` | **FastAPI الرئيسي**: API كامل + SSE للأحداث + رفع ملفات + مصمم أصوات |
| `static/index.html` | الصفحة الرئيسية |
| `static/send.html` | صفحة الرفع |
| `static/projects.html` | صفحة المشاريع |
| `static/studio.html` | مصمم الأصوات |
| `static/app.js` | JavaScript الرئيسي |
| `static/styles.css` | التنسيقات |
| `static/upload.js` | رفع الملفات |
| `static/split.js` | تقسيم الملفات الكبيرة |

### 3. صفحة GitHub Pages (`docs/`)
| الملف | الوظيفة |
|---|---|
| `index.html` | الرئيسية — عرض الدبلجات الجاهزة |
| `send.html` | **إرسال ملف** — رفع مباشر عبر GitHub API |
| `preview.html` | معاينة الفيديو |
| `voices.html` | عرض الأصوات |
| `studio.html` | مصمم الأصوات |
| `github-upload.js` | **محرك الرفع**: تقسيم ملفات كبيرة → blobs → commit واحد |
| `projects.json` | بيانات المشاريع |
| `samples.json` | بيانات العينات الصوتية |
| `samples/` | عينات صوتية (F1-F5, R1-R5) |
| `vendor/` | مكتبات خارجية (lamejs) |
| `*.mp4, *.srt, *.vtt` | الدبلجات الجاهزة |

### 4. المشاريع (`projects/`)
كل مشروع له مجلد مستقل:
```
projects/<اسم>/
├── project.json     ← النص، التوقيتات، الأصوات، الإعدادات
├── source/          ← الفيديو الأصلي
├── voices/          ← تسجيلات لكل شريحة
├── output/          ← الناتج النهائي (mp4 + srt)
└── work/            ← ملفات مؤقتة
```

المشاريع الموجودة:
- `Bob-Proctor-5min-2v` — 52 جملة، 310 ثانية
- `Bob-Proctor-Sample` — 3 جمل، 13 ثانية
- `Phantom-Thread` — 24 جملة، 59 ثانية (دبلجة مصرية)
- `Vikings-Ragnar-Floki` — 5 جمل، 25 ثانية

### 5. السكربتات (`scripts/`)
| الملف | الوظيفة |
|---|---|
| `audition.py` | اختيار الأصوات بقياس F0 |
| `build_demo.py` | بناء عرض تجريبي |
| `build_dub.py` | بناء دبلجة |
| `build_phantom_dub.py` | بناء دبلجة Phantom Thread |
| `clone_project.py` | استنساخ أصوات مشروع |
| `fetch_asr_model.py` | **تحميل موديل Vosk من GitHub** |
| `fetch_inbox.py` | جلب ملفات من inbox |
| `join_parts.py` | **لصق أجزاء الملفات المقسمة** |
| `publish_docs.py` | نشر الدبلجات على GitHub Pages |
| `restore.py` | استعادة نسخة سابقة |
| `retime_from_audio.py` | إعادة توقيت من الصوت |
| `transcribe.py` | تفريغ مستقل |

### 6. الاختبارات (`tests/`)
25+ ملف اختبار يغطون كل المكونات.

### 7. ملفات أخرى
| الملف | الوظيفة |
|---|---|
| `main.py` | نقطة دخول قديمة → `youtube_auto_dub.cli:main` |
| `run_studio.py` | تشغيل الاستوديو → `web.server:main` |
| `pyproject.toml` | إعدادات المشروع + التبعيات |
| `requirements.txt` | المتطلبات |
| `latest_langmap_generate.py` | توليد خريطة اللغات |
| `كيف-أشغل-الاستنساخ.md` | دليل الاستنساخ بالعربية |

---

## 🔗 الروابط المهمة
- **المستودع**: https://github.com/dhiyaddineb-hue/prostudio
- **GitHub Pages**: https://dhiyaddineb-hue.github.io/prostudio/
- **صفحة الإرسال**: https://dhiyaddineb-hue.github.io/prostudio/send.html
- **الفرع**: `arena/01a03969-prostudio`

## ⚙️ القدرات الحالية في البيئة
- ✅ Python 3.11 + PyTorch 2.13 (CPU فقط)
- ✅ Faster-Whisper (مثبت لكن يحتاج تحميل موديل من HuggingFace — محجوب)
- ✅ Vosk + موديل إنجليزي (محمّل من GitHub)
- ✅ PocketSphinx (مثبت، إنجليزي فقط)
- ✅ Edge-TTS (مثبت لكن speech.platform.bing.com محجوب)
- ✅ eSpeak NG (مثبت، يشتغل أوفلاين، صوت روبوتي)
- ✅ TTS (Coqui) + XTTS-v2 (مثبت لكن يحتاج تحميل موديل من HuggingFace — محجوب)
- ✅ Piper TTS (مثبت لكن يحتاج موديل)
- ✅ Praat/parselmouth (مثبت)
- ✅ librosa (مثبت)
- ✅ FFmpeg عبر imageio-ffmpeg
- ❌ GPU غير متاح
- ❌ HuggingFace محجوب (ما يقدر يحمل موديلات Whisper/XTTS/F5)
- ❌ Edge-TTS servers محجوبة

## 📥 الملفات في inbox/
1. `3dde7ebc7c_Instagram.mp4` (6.0 MB)
2. `AQMowxUBqW4_...mp4` (5.0 MB)
3. `AQPHPkqUueg_...mp4` (2.9 MB)
4. `AQPHPkqUueg_...(1).mp4` (2.9 MB) — نسخة مكررة
5. `Phantom Thread...mp4` (3.1 MB)
6. `Do You Know who You Are_ _ Bob Proctor` (4 أجزاء، 61.8 MB)
