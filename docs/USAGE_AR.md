# دليل تشغيل ProStudio للدبلجة العربية إلى الإنجليزية

## 1. نطاق النظام

يقدم هذا المستودع خط إنتاج قابلًا للتشغيل من GitHub Actions يبدأ من ملف فيديو يحتوي على كلام عربي، ثم يستخرج الصوت، ويفصل الكلام عن الخلفية عند تفعيل Demucs، ويُفرّغ الكلام مع التوقيت، ويترجم المقاطع، ويولّد الصوت الإنجليزي عبر VoxCPM، ويطبق Seed-VC لنقل هوية الصوت، ويضبط التوقيت، ثم يعيد تركيب الفيديو ويكتب تقارير قابلة للقراءة الآلية. **النسخة الحالية صوتية فقط؛ مزامنة الشفاه المرئية معطلة عمدًا.**

لا يعتبر نجاح subprocess دليلًا على جودة النتيجة. كل مرحلة يجب أن تكتب حالتها، ومدخلاتها، ومخرجاتها، وتحذيراتها. عند فشل مرحلة بصرية اختيارية، تبقى نسخة الصوت المدبلج هي الناتج الآمن ولا تُحذف.

## 2. التشغيل من GitHub

افتح صفحة [مركز التحكم](dashboard.html) من GitHub Pages للرفع والاختيار والمعاينة والتشخيص. أو افتح تبويب **Actions**، اختر **Run YouTube Auto Dub** ثم **Run workflow** على الفرع `arena/01a03969-prostudio`. اختر `dub` في خانة المهمة، وضع مسار الفيديو داخل المستودع مثل `samples/comingstarr_arabic_source.mp4`. للإعداد المعتاد اختر اللغة المصدر `ar`، اللغة الهدف `en`، محرك `voxcpm`، وفعل `separate_sources` و`bg_music`. الواجهة ترسل دائمًا `lip_sync=false`.

يجب الاحتفاظ بملفات `output/*.mp4` و`output/*.pipeline.json` و`output/*-report.json` كـ artifacts. لا تُرفع المراجع الصوتية أو الأوزان أو السجلات الحساسة إلا عند تفعيل سياسة debug صريحة.

## 3. أسرار GitHub المطلوبة

| الاسم | الاستخدام | السلوك عند الغياب |
|---|---|---|
| `HF_TOKEN` | تحميل pyannote وتشغيل diarization | تُسجّل diarization غير متاحة؛ لا تُخترع هوية متحدث |
| `WAV2LIP_CHECKPOINT` | مسار checkpoint موجود على runner | يُتخطى lip-sync وتبقى الدبلجة الصوتية |
| `WAV2LIP_CHECKPOINT_URL` | رابط HTTPS خاص أو معتمد لتنزيل checkpoint | لا يُستخدم إلا مع checksum |
| `YT_COOKIES` | تنزيل مصادر YouTube عند الحاجة | يستمر التشغيل فقط للمصادر التي لا تحتاج cookies |

عند استخدام `WAV2LIP_CHECKPOINT_URL` يجب إدخال قيمة workflow `wav2lip_sha256`. يقوم الـ workflow بتنزيل الملف مع إعادة المحاولة ثم يرفضه إذا لم يطابق SHA-256. لا تضع token أو checkpoint داخل Git أو داخل ملف YAML.

## 4. ملف الإعداد

الملف الأساسي هو `config/pipeline_profile.yaml`. الوضع `safe` يعطل الفصل والـ diarization ولا يعيد مزج الصوت الأصلي. الوضع `high_quality_single` يستخدم VoxCPM وSeed-VC مع مرجع متحدث واحد. الوضع `multi_speaker_cinematic` يفعّل المراحل الصوتية الاختيارية، لكنها لا تتجاوز بوابات الصلاحية. المزامنة المرئية ليست جزءًا من الاستوديو الحالي.

يمكن تشغيل الأوركسترا محليًا لفحص الإعدادات دون النماذج:

```bash
python pipeline.py \
  --video samples/comingstarr_arabic_source.mp4 \
  --dubbed-audio translated.wav \
  --output output/dub.mp4 \
  --profile config/pipeline_profile.yaml \
  --mode safe \
  --dry-run
```

تستخدم `--set timing.max_tempo_factor=1.10` لتعديل قيمة محددة مؤقتًا. لا تُرفع الحدود الزمنية لمجرد إسكات تحذير؛ إعادة الصياغة أو إعادة التوليد أفضل من ضغط الكلام بشدة.

## 5. قواعد الجودة والمزامنة

تقاس مدة كل مقطع مقابل نافذته الأصلية. يقبل النظام افتراضيًا عامل سرعة بين `0.80` و`1.25`، مع حد مطلق `0.69` إلى `1.45`. لا يجوز قص الكلمة الأخيرة. المقطع الذي لا يلائم نافذته يعاد صياغته أو يولد من جديد أو يوسم للمراجعة اليدوية.

تستخدم بوابة Wav2Lip الآن تغطية الوجه، نسبة مساحة الوجه، نسبة العينات ذات الوجه المفرد، وعدد الوجوه المتعددة. يفشل المسار البصري إذا كانت اللقطات غامضة أو غير ثابتة. لا ينبغي تشغيل الفيديو كاملًا كمسار وجه واحد؛ يجب تقسيمه إلى لقطات أو مسارات مستقرة عندما يضاف متعقب لقطات متقدم.

أي تقرير يذكر `unvalidated` يعني عدم وجود مقياس مزامنة صوتية-بصرية مستقل، ولا يعني أن المزامنة مثالية. ينبغي مراجعة المقاطع التي تحتوي على وجه صغير أو أكثر من وجه أو تمديد زمني كبير.

## 6. استكشاف الأعطال السابقة

إذا رفض VoxCPM المرجع، تحقق من أن المرجع المنظف لا يتجاوز 45 ثانية. إذا فشل Demucs، لا تستخدم المسار الأصلي كخلفية؛ سلّم الحوار المولد وحده أو استخدم fallback غير عصبي موسومًا بوضوح. إذا غاب `HF_TOKEN`، لا تنسب مقطعًا إلى متحدث اعتمادًا على الصوت فقط. لا تُشغّل Wav2Lip في الاستوديو الحالي؛ الناتج المعتمد هو الدبلجة الصوتية فقط.

بعد أي تعديل شغّل:

```bash
python -m compileall -q .
pytest -q
python -m yaml  # أو محلل YAML متوفر في بيئتك
 git diff --check
```

ثم شغّل workflow صغيرًا بميزات محدودة قبل تشغيل فيديو طويل أو وضع سينمائي متعدد المتحدثين. لا تُعلن نجاحًا كاملًا إلا بعد تنزيل artifact وفحص `ffprobe` للمدة، وجود الفيديو، وجود مسار الصوت، وعدم وجود مسار الحوار الأصلي غير المقصود.

## 7. حدود الترخيص والموارد

Wav2Lip مشروع بحثي upstream، ويجب مراجعة شروطه قبل الاستخدام التجاري. كما أن Demucs وpyannote وVoxCPM وSeed-VC تعتمد على أوزان وخدمات قد تتغير أو تتطلب قبول شروط استخدام. يجب تثبيت إصدارات Python والحزم في بيئة CI ومراجعة checkpoints قبل اعتمادها إنتاجيًا.

## المراجع

[1]: https://github.com/dhiyaddineb-hue/prostudio "ProStudio repository"
[2]: https://github.com/pyannote/pyannote-audio "pyannote.audio repository"
[3]: https://github.com/facebookresearch/demucs "Demucs repository"
[4]: https://github.com/Rudrabha/Wav2Lip "Wav2Lip repository"
[5]: https://ffmpeg.org/ffprobe.html "FFprobe documentation"


## 13. الربط الفعلي بين صفحة التحكم وWorkflow

ترسل صفحة `dashboard.html` الآن جميع الخيارات الأساسية إلى Workflow، بما في ذلك `source_lang` و`target_lang` و`voice` و`tts_engine` و`gender` و`model` و`bg_music` و`diarize` و`separate_sources` و`seed_vc`. عند إلغاء خيار الاستنساخ، تكون قيمة `seed_vc=false` ولا تُنفذ مرحلة Seed-VC. وعند تفعيله، ينفذ Workflow `scripts/seed_vc_enhance.py` بعد التوليد لكل محركات الدبلجة التي اختارها المستخدم.

قبل الدبلجة، تنشئ مرحلة `Preflight environment and credentials` الملف `output/preflight.json`. يفحص التقرير وجود FFmpeg وFFprobe، اللغة والمحرك، حالة طلب Seed-VC وDemucs وdiarization، وجود `HF_TOKEN` وcookies، ويثبت أن lip-sync المرئي معطل. إذا اختار المستخدم diarization دون `HF_TOKEN` يتوقف التشغيل قبل استهلاك وقت النماذج ويرفع تقرير التشخيص كـ artifact.

بعد نجاح الدبلجة، ينفذ Workflow `scripts/publish_dub_run.py` لإنشاء مشروع مستقل تحت `projects/`، ويحفظ المصدر والناتج والتقرير والـ manifest، ثم يشغل `scripts/publish_docs.py` لتحديث `docs/projects.json` ونسخ الفيديو وSRT وVTT إلى GitHub Pages. لذلك يظهر كل ناتج ناجح في مكتبة المشاريع والمعاينة بعد اكتمال النشر.
