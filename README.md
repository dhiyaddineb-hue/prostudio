# ProStudio

استوديو جاهز للاستخدام المباشر لدبلجة وترجمة فيديوهات يوتيوب آلياً.

مبني على مشروع [youtube-auto-dub](https://github.com/mangodxd/youtube-auto-dub) مع واجهة ويب عربية، دعم رفع الملفات، وأصوات عربية جاهزة (`حامد` / `سلمى`).

```
رابط يوتيوب أو ملف محلي
        │
        ▼
[1] التنزيل عبر yt-dlp
[2] تفريغ بـ Faster-Whisper + VAD
[3] تقسيم الجمل ≤ 10 ثوانٍ
[4] الترجمة إلى اللغة الهدف
[5] توليد الصوت العصبي Edge-TTS
[6] المطابقة الزمنية atempo + مزج الخلفية
[7] تصدير output.mp4
```

## التشغيل المباشر

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_studio.py
```

ثم افتح الواجهة على المنفذ `8080`.

- الصق رابط يوتيوب **أو** ارفع ملف فيديو
- اللغة الافتراضية: العربية
- الصوت الرجالي الافتراضي: `ar-SA-HamedNeural`
- الصوت النسائي الافتراضي: `ar-EG-SalmaNeural`
- فعّل «الاحتفاظ بموسيقى الخلفية» لمزج المؤثرات الأصلية

### سطر الأوامر

```bash
python -m youtube_auto_dub "https://youtube.com/watch?v=VIDEO_ID" --lang ar --bg-music
python -m youtube_auto_dub video.mp4 -m dub -l ar -g female --voice ar-EG-SalmaNeural
```

## GitHub Actions

انسخ القالب إلى مسار GitHub ثم شغّله يدوياً:

```bash
mkdir -p .github/workflows
cp docs/github-actions/dub.yml .github/workflows/dub.yml
```

1. ارفع المستودع إلى GitHub
2. اذهب إلى **Actions → Run YouTube Auto Dub**
3. ضع الرابط واضغط **Run workflow**
4. حمّل `final-dubbed-video` من Artifacts

الملف الجاهز: `docs/github-actions/dub.yml`

## التشغيل بدون إنترنت

الاستوديو يعمل حتى لو كان الخادم غير متصل أو ينقصه العتاد الثقيل:

| المكوّن | عند توفره | عند غيابه |
| --- | --- | --- |
| `torch` | كشف GPU واستخدام CUDA | التشغيل على CPU مباشرة (اختياري تماماً) |
| `faster-whisper` | تفريغ آلي للصوت | الصق نص الفيديو في خانة النص |
| Edge-TTS | أصوات Microsoft العصبية | أصوات eSpeak NG محلية |
| Google Translate | ترجمة سياقية | محرك ترجمة محلي مدمج |

افحص قدرات الخادم في أي وقت:

```bash
curl localhost:8080/api/health
```

```json
{"ok": true, "ffmpeg": true, "device": "cpu", "whisper": false,
 "edge_tts": false, "espeak": true, "can_dub": true, "needs_transcript": true}
```

تعرض الواجهة هذه الحالة في الشريط العلوي، وتنبّهك إن كان النص مطلوباً.

## الاستنساخ عبر GitHub Actions

الاستنساخ العصبي يحتاج أوزاناً من HuggingFace. إن كانت محجوبة عندك، شغّله على
مشغّلات GitHub التي تصل إليها:

انسخ القالب أولاً إلى مسار GitHub:

```bash
cp docs/github-actions/clone.yml .github/workflows/clone.yml
git add .github/workflows/clone.yml && git commit -m "enable clone workflow" && git push
```

ثم: **Actions → Clone voices and dub → Run workflow**

يقتطع مقطعاً مرجعياً من صوت كل ممثل، يولّد جمله بصوته، ثم يرفع الناتج
كـ artifact. وعلى جهاز يصل إلى HuggingFace:

```bash
pip install f5-tts
PROJECT=<اسم-المشروع> python scripts/clone_project.py
PROJECT=<اسم-المشروع> python scripts/build_phantom_dub.py
```

(`<اسم-المشروع>` مجلد موجود تحت `projects/` يحوي `project.json`؛ التشغيل عبر Actions يرفض قيمة فارغة أو مشروعاً غير موجود.)

التسجيلات السابقة تُنسخ إلى `voices_before_clone/` قبل الاستبدال، فالتراجع ممكن.

> أوزان بعض النماذج مرخّصة **CC BY-NC** (غير تجارية). راجع ترخيص ما تستخدمه.

## تنظيم المشاريع

كل دبلجة لها **مجلد مستقل** تحت `projects/`، مكتفٍ بذاته:

```
projects/<اسم-المشروع>/
├── project.json     النص، التوقيتات، توزيع الأصوات، إعدادات الإخراج
├── source/          الفيديو الأصلي
├── voices/          تسجيل لكل شريحة ترجمة (c01_f.wav …)
├── output/          الناتج النهائي: mp4 + srt
└── work/            ملفات مؤقتة — يمكن حذفها بأمان
```

يمكنك ضغط المجلد أو نقله أو حذفه دون أن يتأثر أي مشروع آخر.

```bash
curl localhost:8080/api/projects          # قائمة المشاريع وتقدّمها
```

المشاهدة: `/watch/p/<اسم-المشروع>`

في Git: يُحفظ `project.json` والتسجيلات والناتج النهائي؛ ويُستبعد الفيديو
المصدر والملفات المؤقتة.

## الاستنساخ الصوتي والدبلجة المتقدمة

عند دبلجة مقطع فيه حوار أصلي، يعمل الخط الأنبوبي على مرحلتين:

1. **عزل الحوار عن الموسيقى** — `youtube_auto_dub/stem_split.py` يفصل القناة
   المركزية (الحوار) عن الجوانب (الموسيقى) بمعالجة إشارة خالصة، بلا أي نموذج.
2. **نقل هوية الصوت** — بأفضل وسيلة متاحة:

| المتاح | الطريقة |
| --- | --- |
| أوزان F5-TTS | استنساخ عصبي حقيقي من صوت الممثل الأصلي |
| بدونها | مطابقة صوتية: النبرة + الصيغ الرنينية عبر Praat |

الاستنساخ يُفعَّل تلقائياً بمجرد توفر الأوزان:

```bash
pip install f5-tts
python scripts/build_phantom_dub.py
```

يطبع السكربت `neural cloning: ENABLED` عند نجاح تحميل الأوزان، وإلا يتراجع
إلى المطابقة الصوتية دون أن يفشل.

## المتطلبات

- Python 3.10+
- FFmpeg (أو الحزمة `imageio-ffmpeg` المضمّنة)
- `torch` و`faster-whisper` اختياريان — للتفريغ الآلي وتسريع GPU
- اتصال إنترنت اختياري: للترجمة السياقية وEdge-TTS وتحميل يوتيوب

### مصدر الفيديو

الأولوية دائماً لما تحدّده صراحةً — الرابط أو المسار أو الملف المرفوع.
مجلد `inbox/` يُستخدم فقط عند تشغيل الأمر بلا مصدر.

## الاستوديو على الويب

الصفحة الرئيسية للموقع <https://dhiyaddineb-hue.github.io/prostudio/> هي **الاستوديو الكامل**:
مكتبة الفيديوهات المرفوعة (كل فيديو في مجلده `library/<الاسم>/`)، النسخ المدبلجة
(`dubs/<الاسم>/final-dub-<run>.mp4` مع تقاريرها)، معاينة، رفع (مع تقسيم تلقائي للملفات
الكبيرة)، تشغيل الدبلجة، متابعة التشغيلات مباشرةً (الخطوة الحالية وتقدّم المقاطع من نقاط
الاستئناف)، وحذف بتأكيد صريح (كتابة اسم المجلد). الصفحة ثابتة وتتعامل مع GitHub API
من المتصفح برمز شخصي يُحفَظ محلياً؛ أي فيديو يُرفع أو يُدبلج يظهر تلقائياً عند التحديث الدوري.

## بنية المستودع الرسمية

نُظّف المستودع في 2026-09-06 من كل ملفات التجارب القديمة (فيديوهات Napoleon والمقاطع
التجريبية والمخرجات المنشورة السابقة، ~730 MB، إضافة إلى checkpoints بحجم ~1 GB).
البنية المعتمدة للعمل الرسمي:

```
youtube_auto_dub/   النواة: تفريغ، ترجمة (Google أو LLM اختياري)، TTS، مزامنة، مزج
scripts/            نقاط التشغيل: resumable_smart_dub.py (الخط الرسمي)، publish_*، validate_*، أدوات مساعدة
tests/              اختبارات pytest (بلا شبكة)
web/ · studio/      الواجهة المحلية والاستوديو
lip_sync/ · config/ مزامنة الشفاه (اختياري) وملفات الإعداد
.github/workflows/  dub.yml (الدبلجة الرسمية)، translation-preflight.yml، cleanup-dub-checkpoints.yml، …
docs/               موقع GitHub Pages: الاستوديو (index.html + studio.js)، الأدلة (guides/)، أبحاث (research/)
library/            الفيديوهات المرفوعة: مجلد لكل فيديو (source.mp4 أو أجزاء .partNNofMM + meta.json)
dubs/               الفيديوهات المدبلجة: مجلد لكل فيديو، نسخة لكل تشغيل + تقارير + meta.json
samples/            عيّنتان صغيرتان فقط للتجربة (ProStudio_Arabic_Demo.mp4 و shorts-test.mp4) + أصوات الاستوديو
projects/           مشاريع الاستنساخ (clone) فقط، تُنشأ يدوياً
inbox/              رفعات مؤقتة — لا تُحفَظ في git
```

**قواعد ثابتة**: لا حذف لملفات أو checkpoints دون موافقة صريحة (سير العمل
`cleanup-dub-checkpoints.yml` يطلب كتابة `DELETE <project_id>`)؛ المصادر الكبيرة لا
تُرفع إلى git إلا عند الحاجة وكأجزاء `.part0/.part1` أقل من 100 MB؛ والتفاصيل التقنية
اليومية في `PROJECT_NOTES.md`.

## الترخيص

MIT. نواة الخط الأنبوبي من youtube-auto-dub © Nguyen Cong Thuan Huy.

## دليل خط الإنتاج النهائي

للتشغيل الإنتاجي من GitHub، وإعداد الأسرار، وفحص checkpoints، وقواعد المزامنة
والـ fallback، راجع [دليل الاستخدام العربي الشامل](docs/USAGE_AR.md). كما يتوفر
[مخطط مراحل خط الإنتاج](docs/pipeline-flow-ar.png) وملف Mermaid القابل للتعديل
`docs/pipeline-flow-ar.mmd`.
