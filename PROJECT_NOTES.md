# مذكرة المشروع — ProStudio

> ## ⚠️ قواعد ثابتة من المستخدم — اقرأها أولاً في كل جلسة
> 1. **الفرع الأساسي للمستخدم هو `arena/01a03969-prostudio` دائماً** — كل العمل يجب أن يصل إليه في النهاية.
> 2. **GitHub Pages يعمل من هذا الفرع تحديداً** (المسار `/docs`) — أي تعديل على الموقع يجب أن يصل لهذا الفرع حتى يظهر.
> 3. **حقيقة تقنية لا يمكن تجاوزها**: كل جلسة أرينا تُنشئ **فرع جلسة** مرتبطاً بالمنصة نفسها
>    (الجلسة الحالية مربوطة بـ `arena/01a04ec9-prostudio`)، ولا يستطيع الوكيل تحويل
>    الجلسة إلى فرع آخر أو الدفع إلى فرع غير فرع جلسته. هذا قيد من المنصة وليس تفضيلاً.
> 4. **سير العمل الإلزامي** (وهو الطريقة الوحيدة للوصول لفرع المستخدم):
>    `git fetch` ← `git merge --ff-only origin/arena/01a03969-prostudio` ← العمل ← الدفع لفرع الجلسة
>    ← فتح PR إلى `arena/01a03969-prostudio` ← الدمج ← Pages يعيد البناء تلقائياً.
>    مثال سابق: **PR #4** من `arena/01a04ec2-prostudio` دُمج في 2026-08-29 18:24.
> 5. المستخدم يعتبر GitHub هو المشغّل والبيئة، والوكيل هو العقل الذي يصدر الأوامر له.

## الفرع: arena/01a03969-prostudio (فرع المستخدم الدائم)

### حالة التحقق الأخيرة — 2026-08-29

| البند | القيمة المُخرَجة |
|---|---|
| المستودع | `https://github.com/dhiyaddineb-hue/prostudio` |
| فرع الجلسة (مربوط بالمنصة) | `arena/01a04ec9-prostudio` |
| فرع المستخدم | `origin/arena/01a03969-prostudio` |
| رأس فرع المستخدم | **`ea677a7`** — «ci: تثبيت Whisper وتشغيل الدبلجة داخل GitHub» (2026-08-29 18:31:50) |
| الكوميت السابق | `0025771` — «fix: لا تستخدم التفريغ الإنجليزي للكشف التلقائي» |
| `main` | `f63835f` — «Create dub.yml» |
| علاقة فرع الجلسة بفرع المستخدم | **متطابقان** — فرع الجلسة مُمرَّر fast-forward إلى `ea677a7` (`git rev-list --left-right --count` = `0  0` بعد الدمج) |
| GitHub Pages | `https://dhiyaddineb-hue.github.io/prostudio/` — الحالة `built` |
| مصدر Pages | الفرع `arena/01a03969-prostudio`، المسار `/docs`، `build_type: legacy`، HTTPS مفروض |
| آخر بناء Pages | من `ea677a7` في `2026-08-29T18:32:34Z` — ناجح، مدته 41 ثانية في البناء السابق |
| صفحة الإرسال | `https://dhiyaddineb-hue.github.io/prostudio/send.html` |

> ⚠️ **تحقّق دائماً بـ `git ls-remote --heads origin`** — فرع المستخدم يتحرك بين
> الجلسات. في مراجعة 2026-08-29 تبيّن أنه تقدّم من `0025771` إلى `ea677a7` أثناء
> الجلسة نفسها، فدُمج fast-forward قبل أي تعديل.

- ملاحظة: المشروع الحالي **ProStudio للدبلجة والترجمة** وليس مشروع لعبة؛ لا يوجد وصف لعبة محدد في الطلب الحالي، لذلك لم تُنشأ لعبة أو بنية Babylon/WebDev اعتباطية.

### ✅ نتيجة الفحص الفعلي — 2026-08-29

```
$ .venv/bin/python -m pytest -q
171 passed, 3 skipped, 0 failed        (5.29s)

$ for f in tests/browser/*.mjs; do node "$f"; done
github-upload 32 · shrink 12 · split 26 · voice-design 36  → 106 تأكيداً، 0 فشل
```

- اختبارات `.mjs` سكربتات قائمة بذاتها: `node tests/browser/shrink.test.mjs`
  (وليس `node --test tests/browser/` — هذا يفشل).
- الاختبارات تحتاج بيئة افتراضية: بايثون النظام هنا محكوم بـ PEP 668.
  ```
  python3 -m venv .venv
  .venv/bin/python -m pip install pytest pytest-asyncio numpy fastapi uvicorn \
      python-multipart aiofiles httpx rich pydub beautifulsoup4 yt-dlp soundfile \
      imageio-ffmpeg requests edge-tts librosa praat-parselmouth espeakng-loader
  ```
  بدون `edge-tts` و`librosa` و`praat-parselmouth` يفشل **جمع** 5 ملفات اختبار.
- **عُيِّن وإصلاح**: `tests/test_pipeline_args.py::test_build_args_arabic_source_english_dub`
  كان فاشلاً لأنه يتوقع `args.lang_dub == "en"` بينما `build_args` يبقيه `None`
  عمداً؛ الحسم يحدث في `core.py:53` بـ `dub_lang = args.lang_dub or base_lang`،
  و`core.py:310` يعتمد على كونه `None` لإضافة لاحقة `_D-<lang>` لاسم الناتج.
  عُدِّل الاختبار ليصف العقد الحقيقي، وأُضيف `test_build_args_explicit_dub_lang_is_kept`.

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
| `speech_windows.py` | **نوافذ الكلام الفعلية** — يقيس متى يتكلم الممثل من القناة المركزية بدل الاعتماد على مدة الترجمة الظاهرة (كانت أوسع من حركة الفم: 1.50s مقابل 1.14s في مقطع Vikings) |
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
| `upload.js` | رفع بسيط للملفات |
| `split.js` | تقسيم الملفات الكبيرة في المتصفح |
| `shrink.js` | ضغط الملفات قبل الرفع |
| `voice-design.js` | منطق مصمم الأصوات |
| `github-actions/` | قوالب workflows — **يوجد `ci.yml` و`dub.yml` فقط** |
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
| `build_pro_demo.py` | بناء عرض تجريبي احترافي |
| `clone_project.py` | استنساخ أصوات مشروع |
| `fetch_asr_model.py` | **تحميل موديل Vosk من GitHub** |
| `fetch_inbox.py` | جلب ملفات من inbox |
| `join_parts.py` | **لصق أجزاء الملفات المقسمة** |
| `publish_docs.py` | نشر الدبلجات على GitHub Pages |
| `restore.py` | استعادة نسخة سابقة |
| `retime_from_audio.py` | إعادة توقيت من الصوت |
| `transcribe.py` | تفريغ مستقل |

### 6. الاختبارات (`tests/`)
**31 ملفاً** (مُخرج من `git ls-files tests`): 26 وحدة اختبار Python + `__init__.py`
+ 4 ملفات `.mjs` تحت `tests/browser/` تختبر منطق `docs/` مباشرة.

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
- **صفحة الإرسال (رفع ملف)**: https://dhiyaddineb-hue.github.io/prostudio/send.html
- **فرع المستخدم على GitHub**: https://github.com/dhiyaddineb-hue/prostudio/tree/arena/01a03969-prostudio
- **الفرع**: `arena/01a03969-prostudio` (المصدر الرسمي لـ Pages من مسار `/docs`)

## ⚙️ القدرات الحالية في البيئة

> **تصحيح مهم (2026-08-29)**: الجدول القديم في هذا القسم كان يصف **صندوق رمل سابقاً**
> وقد تحقّق أنه لم يعد صحيحاً هنا. بايثون النظام كان فيه 3 حزم فقط (`pip`,
> `setuptools`, `wheel`) ولا `.venv` قبل هذه الجلسة. الأرقام أدناه هي **مُخرج فعلي**
> من مسبار المشروع نفسه `youtube_auto_dub.runtime.capabilities()` في هذا الصندوق:

```json
{"ffmpeg": true, "device": "cpu", "torch": false, "whisper": false,
 "whisper_installed": false, "whisper_models_cached": [], "model_hub": false,
 "offline_asr": false, "edge_tts": false, "espeak": true, "studio_voices": 5,
 "can_dub": true, "needs_transcript": true}
```

| المكوّن | الحالة هنا | الدليل |
|---|---|---|
| Python | 3.11.2 | `python3 -V` |
| FFmpeg | ✅ متاح | `ffmpeg_bin.ffmpeg_exe()` عبر `imageio-ffmpeg` |
| GPU | ❌ CPU فقط | `pick_device() == "cpu"` |
| `torch` | ❌ **غير مثبّت** | `have_module("torch") → False` |
| Faster-Whisper | ❌ **غير مثبّت** | `whisper_installed: false` |
| Vosk / PocketSphinx | ❌ **غير مثبّتة** | `have_module → False`، لذا `offline_asr: false` |
| Piper / Coqui (XTTS-v2) | ❌ **غير مثبّتة** | `have_module("piper"/"TTS") → False` |
| Edge-TTS | ⚠️ الحزمة مثبّتة **لكن الخادم محجوب** | `speech.platform.bing.com` يفشل مصافحة TLS |
| eSpeak NG | ✅ يعمل أوفلاين (صوت روبوتي) | `espeak: true` |
| أصوات الاستوديو الجاهزة | ✅ **5 مقاطع** في `samples/voices/` | `studio_takes() == 5` |
| librosa / praat-parselmouth | ✅ مثبّتان (ثبّتهما الوكيل للتشغيل) | `have_module → True` |

**قواعد الاستدلال في `runtime.capabilities()`** (مقروءة من الشيفرة، لا مُخمَّنة):
- `whisper` = الحزمة مثبّتة **و** (أوزان مخزّنة محلياً **أو** `huggingface.co` قابل للوصول).
- `edge_tts` = **قابلية وصول** `speech.platform.bing.com`، لا مجرد تثبيت الحزمة.
- `can_dub = ffmpeg and (edge_tts or espeak or studio_voices > 0)` → `true` هنا.
- `needs_transcript = not (whisper or offline_asr)` → `true` هنا، أي أن **الصق النص مطلوب**.

**فحص الشبكة** (`host_reachable` يُكمل مصافحة TLS حقيقية):

| المضيف | النتيجة |
|---|---|
| `github.com:443` | ✅ متاح |
| `pypi.org:443` | ✅ متاح (لذلك أمكن `pip install`) |
| `huggingface.co:443` | ❌ محجوب → لا Whisper ولا XTTS ولا F5 |
| `speech.platform.bing.com:443` | ❌ محجوب → Edge-TTS لا يعمل |
| `dhiyaddineb-hue.github.io:443` | ❌ محجوب من الصندوق (`curl` يفشل بـ `SSL_ERROR_SYSCALL`) — **الموقع نفسه سليم**؛ تحقّق من خارج الصندوق أرجع HTTP 200 ومحتوى الصفحة |

> **النتيجة العملية**: كل تفريغ آلي معطّل هنا، فالدبلجة تحتاج نصاً ملصوقاً،
> والصوت يُولَّد محلياً بـ eSpeak أو من مقاطع `samples/voices/`. أما الاستنساخ
> العصبي (F5/XTTS) فيحتاج أوزاناً من HuggingFace → يُشغَّل على **مشغّلات GitHub**
> عبر `.github/workflows/dub.yml`، وهذا هو سبب وجوده أصلاً.

## 📥 الملفات في inbox/
1. `3dde7ebc7c_Instagram.mp4` (6.0 MB)
2. `AQMowxUBqW4_...mp4` (5.0 MB)
3. `AQPHPkqUueg_...mp4` (2.9 MB)
4. `AQPHPkqUueg_...(1).mp4` (2.9 MB) — نسخة مكررة
5. `Phantom Thread...mp4` (3.1 MB)
6. `Do You Know who You Are_ _ Bob Proctor` (4 أجزاء، 61.8 MB)

كلها **متتبَّعة في Git** (`git ls-files inbox` = 10 مدخلات مع `.gitkeep`)، أي أن
المستودع يحمل ~85 ميغابايت من الفيديو الخام. هذا مقصود في التصميم الحالي لأن
`scripts/fetch_inbox.py` و`scripts/join_parts.py` يعتمدان على وصولها من GitHub.

---

## 🔎 ملاحظات مفتوحة اكتُشفت في مراجعة 2026-08-29

1. **`README.md:92-93` يشير إلى ملف غير موجود**: يأمر بـ
   `cp docs/github-actions/clone.yml .github/workflows/clone.yml`، بينما المجلد
   يحتوي `ci.yml` و`dub.yml` فقط. النسخ **سيفشل**. لكن `dub.yml` نفسه يضم مهمة
   `clone` داخله (`inputs.task: clone`)، فالأمر الصحيح هو تشغيل
   **Actions → Run YouTube Auto Dub → task: clone** لا نسخ ملف مستقل.
   لم يُعدَّل الـREADME لأنه توثيق سلوك، ويحتاج قراراً من المستخدم.
2. **`docs/github-actions/dub.yml` نسخة قديمة** — التعديلات التي أضافها المستخدم في
   `ea677a7` (مدخلات `source_path` و`source_lang` وتحميل أوزان Whisper) طُبِّقت على
   `.github/workflows/dub.yml` **فقط**، لا على القالب في `docs/`. أي مستخدم ينسخ
   القالب من `docs/` سيحصل على النسخة القديمة.
3. **`web/app.py:317` يستخدم `@app.on_event("startup")`** المهجور في FastAPI؛
   يظهر كتحذير `DeprecationWarning` عند كل تشغيل للاختبارات.
4. **`ci.yml` يثبّت `.[dev]` فقط، و`librosa` ناقص من `pyproject.toml`** —
   تحقّق تجريبي (2026-08-29): بإزالة `librosa` فعلياً من البيئة (وليس مجرد
   `pip uninstall` الذي يخلّف مجلد `librosa` بمسار namespace فارغ) يفشل اختباران:
   ```
   FAILED tests/test_diarize.py::test_training_separates_two_distinct_voices - ModuleNotFoundError
   FAILED tests/test_diarize.py::test_too_few_seeds_is_refused              - ModuleNotFoundError
   2 failed, 169 passed, 3 skipped
   ```
   `librosa` يُستورد بتأخير داخل `youtube_auto_dub/diarize.py:57` و`audio.py:304`،
   وهو **غير مذكور** في `[project.dependencies]` (المذكور فيها: faster-whisper,
   torch, yt-dlp, pydub, edge-tts, httpx, beautifulsoup4, numpy, rich, soundfile,
   imageio-ffmpeg, fastapi, uvicorn, python-multipart, aiofiles, espeakng-loader,
   praat-parselmouth, pocketsphinx, piper-tts). لذا `pip install .[dev]` في CI
   **سيفشل بهذين الاختبارين**. الإصلاح: إضافة `librosa>=0.10` إلى `dependencies`
   أو `pip install -r requirements.txt` في `ci.yml`.
   (تصحيح لنسخة سابقة من هذه المذكرة: `edge-tts` **موجود** في التبعيات، فلم يكن هو المشكلة.)
