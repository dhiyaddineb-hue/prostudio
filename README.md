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
PROJECT=Phantom-Thread python scripts/clone_project.py
PROJECT=Phantom-Thread python scripts/build_phantom_dub.py
```

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

## الترخيص

MIT. نواة الخط الأنبوبي من youtube-auto-dub © Nguyen Cong Thuan Huy.

## دليل خط الإنتاج النهائي

للتشغيل الإنتاجي من GitHub، وإعداد الأسرار، وفحص checkpoints، وقواعد المزامنة
والـ fallback، راجع [دليل الاستخدام العربي الشامل](docs/USAGE_AR.md). كما يتوفر
[مخطط مراحل خط الإنتاج](docs/pipeline-flow-ar.png) وملف Mermaid القابل للتعديل
`docs/pipeline-flow-ar.mmd`.
