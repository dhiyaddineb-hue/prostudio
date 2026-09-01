# سجل محادثة وتنفيذ مشروع ProStudio

**التاريخ:** 30 أغسطس 2026  
**المستودع:** `dhiyaddineb-hue/prostudio`  
**الفرع:** `arena/01a03969-prostudio`

## 1. الهدف الأصلي

طلب المستخدم بناء خط إنتاج احترافي وقابل للتخصيص لدبلجة الفيديو من العربية إلى الإنجليزية، مع الحفاظ على جودة الصوت واستنساخ هوية المتحدث، ودعم تعدد المتحدثين، وفصل الحوار عن الموسيقى والمؤثرات، وتحسين التوقيت، وإمكانية مزامنة حركة الشفاه، وإخراج فيديو MP4 وتقرير جودة JSON.

كان ترتيب خط الإنتاج المطلوب:

> الفيديو الأصلي → استخراج وفحص الصوت → فصل المصادر اختياريًا → تحديد المتحدثين → التفريغ والتوقيت → الترجمة وتكييف الحوار → توليد واستنساخ الصوت → ضبط التوقيت → مزج الخلفية المعزولة فقط → مزامنة الشفاه اختياريًا → فحص الجودة → MP4 وتقرير JSON.

كما طلب المستخدم الحفاظ على المسار الموجود داخل المستودع، خصوصًا مسار VoxCPM ثم Seed-VC، وعدم اعتبار أي نموذج ناجحًا عند غياب checkpoint أو token أو عند فشل اختبار الجودة.

## 2. المستودع والفرع

تم فحص المستودع واستُخدم الفرع المطلوب فقط:

```text
arena/01a03969-prostudio
```

المكونات الرئيسية التي تم العثور عليها واستخدامها:

- `pipeline.py` كنقطة orchestration.
- `scripts/full_dubbing_pipeline.py` كمسار دبلجة شامل سابق.
- `scripts/seed_vc_enhance.py` لتشغيل Seed-VC وإعادة تركيب الخلفية.
- `youtube_auto_dub/voxcpm_tts.py` لتوليد الصوت عبر VoxCPM.
- `youtube_auto_dub/source_separation.py` لتكامل Demucs.
- `youtube_auto_dub/speaker_diarization.py` لتكامل pyannote.
- `lip_sync/run.py` لتشغيل Wav2Lip المشروط.
- `.github/workflows/dub.yml` لتشغيل خط الإنتاج من GitHub Actions.

## 3. التعديلات الأولى

تم إنشاء ملف الإعداد الرئيسي:

- `config/pipeline_profile.yaml`

ويحتوي على أوضاع:

- `safe`
- `high_quality_single`
- `multi_speaker_cinematic`
- `experiment`

ويتحكم في اللغات، محرك TTS، Seed-VC، فصل المصادر، diarization، التوقيت، lip-sync، حدود الجودة، وسياسات الرجوع الآمن.

تمت إعادة بناء `pipeline.py` ليشمل:

- تحميل profile ودمج إعدادات mode.
- دعم `--set key=value`.
- وضع `--dry-run`.
- تقارير مراحل منظمة.
- فحص FFmpeg وFFprobe.
- منع إعادة مزج الحوار الأصلي.
- فحص وجود الفيديو ومسار الصوت والمدة.
- تسجيل التحذيرات وعمليات fallback.

تم تحسين `source_separation.py` ليشمل:

- اختيار نموذج Demucs قابلًا للضبط.
- التحقق من وجود stem الكلام وstem الخلفية.
- فحص المدة والطاقة والملف.
- إبقاء contamination على حالة `unknown` بدل الادعاء بأن الفصل كامل.

تم تحسين `speaker_diarization.py` ليشمل:

- النموذج الافتراضي `pyannote/speaker-diarization-community-1`.
- حد تغطية قابلًا للضبط، افتراضيًا `0.55`.
- عدم اختراع دور متحدث يغطي الفيديو كاملًا.
- إرجاع المقاطع دون speaker عند عدم كفاية الدليل.

## 4. أول اختبار محلي

نجح ما يلي:

- Python compilation.
- فحص YAML للـ profile والـ workflow.
- dry-run للحالة الآمنة.
- الاختبارات المركزة الأولية: `18 passed`.

وصلت مجموعة الاختبارات الكاملة إلى `161 passed` مع خمس حالات فشل بيئية مرتبطة باعتماد `espeak-ng` ودعم asyncio، وليست بسبب التعديلات الجديدة.

## 5. رفع الفيديو وتشغيل GitHub Actions

تم رفع الفيديو المرفق إلى:

```text
samples/comingstarr_arabic_source.mp4
```

ثم تم تشغيل Workflow بإعدادات:

- المصدر: العربية.
- الهدف: الإنجليزية.
- VoxCPM.
- Seed-VC.
- Demucs.
- pyannote.
- الحفاظ على الخلفية.
- Wav2Lip.
- `multi_speaker_cinematic`.
- فحص `strict`.

### التشغيل الأول

فشل التشغيل الأول في مرحلة توليد VoxCPM لأن المرجع الصوتي كان بطول الفيديو كاملًا، حوالي 99 ثانية، بينما واجهة VoxCPM تقبل مرجعًا لا يتجاوز 50 ثانية.

### الإصلاح

تم تعديل `youtube_auto_dub/core.py` ليقوم قبل استدعاء VoxCPM بـ:

- تنظيف الصوت المرجعي.
- تحويله إلى mono بمعدل 22050 Hz.
- تحديده إلى 45 ثانية.
- استخدام المرجع المنظف نفسه لكل المقاطع.

Commit الإصلاح:

```text
224dc89 Bound and clean VoxCPM reference audio
```

### التشغيل الثاني

نجح VoxCPM وSeed-VC، لكن مرحلة Wav2Lip فشلت لأن السر التالي كان فارغًا:

```text
WAV2LIP_CHECKPOINT
```

### الإصلاح الثاني

تم تعديل `.github/workflows/dub.yml` بحيث:

- لا يحذف نتيجة Seed-VC عند غياب checkpoint.
- يكتب `lip-sync-report.json`.
- يرفع الدبلجة الصوتية مع تحذير واضح.
- يدعم `WAV2LIP_CHECKPOINT_URL`.
- يفرض SHA-256 عند تنزيل checkpoint.

Commit الإصلاح:

```text
28af70f Preserve Seed-VC dub when lip-sync checkpoint is absent
```

## 6. التشغيل الناجح

تم تشغيل Workflow بنجاح في:

[GitHub Actions Run 33306846954](https://github.com/dhiyaddineb-hue/prostudio/actions/runs/33306846954)

المراحل التي اكتملت:

- استخراج الصوت.
- Whisper والتفريغ.
- الترجمة.
- VoxCPM.
- Seed-VC.
- Demucs.
- إعادة تركيب الفيديو.
- رفع artifact.

تم تنزيل artifact والتحقق منه. الناتج الأساسي كان:

```text
seedvc-enhanced.mp4
```

المواصفات التي تم التحقق منها:

| الفحص | النتيجة |
|---|---|
| مدة المصدر | 98.869 ثانية |
| مدة الناتج | 98.930 ثانية |
| فرق المدة | 0.061 ثانية تقريبًا |
| الفيديو | H.264 |
| الصوت | AAC، قناة واحدة |
| وجود مسار صوتي | نعم |
| إعادة مزج الحوار العربي الأصلي | لا |
| VoxCPM | نجح |
| Seed-VC | نجح |
| Wav2Lip | لم يطبق بسبب غياب checkpoint |

النتيجة المحلية:

- [comingstarr_full_pipeline_english.mp4](/home/ubuntu/comingstarr_full_pipeline_english.mp4)
- [comingstarr_full_pipeline_english.pipeline.json](/home/ubuntu/comingstarr_full_pipeline_english.pipeline.json)
- [lip-sync-report.json](/home/ubuntu/full_dub_artifact/lip-sync-report.json)

## 7. تحسين المزامنة والتثبيت النهائي

طلب المستخدم تحسين دقة المزامنة وحل المشاكل السابقة وجعل المشروع يعمل من GitHub دون مشاكل.

تم تحديث `lip_sync/run.py` ليشمل:

- فحص ظهور الوجه في الإطارات.
- قياس نسبة وجود وجه مفرد.
- قياس مساحة الوجه.
- رصد الإطارات متعددة الوجوه.
- رفض الحالات الغامضة.
- منع عوامل السرعة غير الآمنة.
- التحقق من وجود ناتج Wav2Lip فعليًا.
- استخدام المسار الصوتي المدبلج فقط.
- mux آمن باستخدام ملف مؤقت.
- الاحتفاظ بنسخة audio-only عند فشل المرحلة البصرية.

تم تحديث Workflow ليشمل:

- `lip_sync_backend`.
- `profile`.
- `quality`.
- `wav2lip_sha256`.
- تنزيل checkpoint مع checksum.
- تسجيل سبب تخطي lip-sync بدل فشل المهمة كلها.

تم إنشاء:

- `docs/USAGE_AR.md` — دليل الاستخدام العربي الشامل.
- `tests/test_lipsync_safety.py` — اختبارات أمان التوقيت والمزامنة.
- رابط الدليل داخل `README.md`.

نتائج الاختبار بعد التحديث:

```text
26 passed
```

كما نجحت:

- compilation.
- YAML validation.
- `git diff --check`.

Commit التثبيت النهائي:

```text
2d5c27eb653ff76dff4eb0c79233ca8a2d461916
```

## 8. المخطط العربي

تم إنشاء مخطط خط الإنتاج بصيغتين:

- `docs/pipeline-flow-ar.mmd`
- `docs/pipeline-flow-ar.png`

Commit المخطط:

```text
ce8d084 Add Arabic dubbing pipeline flowchart
```

## 9. الحالة النهائية

المشروع يعمل من GitHub Actions لمسار الدبلجة الصوتية الكامل، ويتطلب الميزات الاختيارية التالية لتعمل دون تخطٍ:

| المتغير أو الملف | الغرض |
|---|---|
| `HF_TOKEN` | تشغيل pyannote diarization الحقيقي |
| `WAV2LIP_CHECKPOINT` | تشغيل Wav2Lip من مسار محلي على runner |
| `WAV2LIP_CHECKPOINT_URL` | تنزيل checkpoint عبر HTTPS |
| `wav2lip_sha256` | التحقق من سلامة checkpoint المنزّل |
| `YT_COOKIES` | الوصول إلى مصادر YouTube التي تتطلب cookies |

عند غياب `HF_TOKEN` لا يخترع النظام متحدثين. وعند غياب checkpoint الخاص بـ Wav2Lip لا يحذف ناتج VoxCPM وSeed-VC، بل يخرج الدبلجة الصوتية مع تقرير يوضح أن lip-sync لم يُطبق.

## 10. روابط المستودع والوثائق

- [المستودع والفرع المطلوب](https://github.com/dhiyaddineb-hue/prostudio/tree/arena/01a03969-prostudio)
- [دليل الاستخدام العربي](USAGE_AR.md)
- [مخطط خط الإنتاج](pipeline-flow-ar.png)
- [ملف profile](../config/pipeline_profile.yaml)
- [Workflow](../.github/workflows/dub.yml)
- [تشغيل GitHub Actions الناجح](https://github.com/dhiyaddineb-hue/prostudio/actions/runs/33306846954)

## 11. ملاحظات صريحة

المسار الصوتي الكامل الذي تم اختباره استخدم VoxCPM وSeed-VC بنجاح. أما Wav2Lip فلم يُطبق في التشغيل النهائي بسبب غياب checkpoint موثق في GitHub Secrets. كما أن سجل التشغيل أظهر غياب `HF_TOKEN`، ولذلك لا يمكن اعتبار diarization متعدد المتحدثين معتمدًا في تلك الجولة.

كذلك، يجب عدم وصف أي نتيجة بأنها مثالية أو متزامنة بشكل كامل دون مراجعة بشرية ومقياس صوتي-بصري مستقل. النظام الحالي مصمم ليكون صريحًا بشأن حدود النماذج والاعتمادات، ويقدم أفضل نتيجة آمنة بدل الادعاء بنجاح غير مثبت.

## 12. التسلسل الزمني للقرارات

1. فحص المستودع والفرع والملفات الموجودة.
2. إنشاء profile قابل للتخصيص.
3. إعادة بناء الأوركسترا والتقارير.
4. تحسين فصل المصادر وdiarization.
5. رفع فيديو الاختبار إلى الفرع.
6. تشغيل Workflow كامل.
7. إصلاح حد مرجع VoxCPM إلى 45 ثانية.
8. إعادة التشغيل ونجاح VoxCPM وSeed-VC.
9. اكتشاف غياب Wav2Lip checkpoint.
10. تعديل Workflow ليحتفظ بالناتج الآمن.
11. تشغيل Workflow بنجاح.
12. تنزيل artifact والتحقق من MP4.
13. تحسين بوابة الوجه والتوقيت والـ retry behavior.
14. إضافة التوثيق العربي والاختبارات.
15. تثبيت commit النهائي على الفرع.


## 13. مركز التحكم الصوتي داخل GitHub Pages

طلب المستخدم تحويل المشروع إلى استوديو كامل داخل GitHub، مع رفع الملفات، اختيار اللغات والمحرك، التحكم في استنساخ الصوت، الاحتفاظ بالموسيقى والمؤثرات، إنتاج الصوت فقط، المعاينة، والتشخيص، مع إلغاء المزامنة المرئية للشفاه.

تم إنشاء صفحة `docs/dashboard.html` وتربط مباشرة بفرع `arena/01a03969-prostudio`. الصفحة توفر:

- رفع ملف فيديو أو صوت إلى مجلد `inbox/` عبر GitHub Git Data API.
- اختيار لغة المصدر والهدف.
- اختيار VoxCPM أو XTTS-v2 أو Qwen أو Edge-TTS.
- اختيار الجنس الافتراضي ونموذج Whisper وسياسة التوقيت.
- تفعيل أو تعطيل Seed-VC ونقل هوية الصوت.
- تفعيل أو تعطيل Demucs والحفاظ على الموسيقى والمؤثرات.
- خيار diarization عند توفر `HF_TOKEN`.
- تشغيل Workflow تلقائيًا بعد الرفع.
- مكتبة ديناميكية تقرأ كل ملفات الفيديو من `samples/` و`inbox/` و`output/` في الفرع.
- معاينة الفيديو داخل الصفحة وروابط التحميل وفتح الملف في GitHub.
- أداة تشخيص تقرأ حالة المستودع وآخر تشغيل Actions، مع التصريح بأن Secrets لا يمكن قراءتها من API.
- عدم عرض أو تشغيل خيار lip-sync المرئي، وإرسال `lip_sync=false` دائمًا.

تم تعطيل خطوة lip-sync المرئي في Workflow نفسه (`if: false`) حتى لا يتم تشغيل Wav2Lip من التشغيل العادي، وأصبح الناتج النهائي صوتيًا فقط مع مزج الخلفية عند اختيارها.

تمت مراجعة الحالة الحالية للمستودع، وتبين وجود تشغيلات ناجحة وملفات دبلجة حديثة داخل `output/dubbed/`، إضافة إلى ملفات مصدر كثيرة داخل `samples/`. وبسبب طلب المستخدم الحفاظ على الكوتا وعدم وجود فيديو جديد مرفق في هذه الجولة، تم الاعتماد على تشغيل الدبلجة الناجح السابق بدل استهلاك كوتا إضافية بتشغيل مطابق.

Commit صفحة التحكم:

```text
ac849d9 Add audio dubbing control studio and repository media library
```

رابط GitHub Pages:

https://dhiyaddineb-hue.github.io/prostudio/dashboard.html
