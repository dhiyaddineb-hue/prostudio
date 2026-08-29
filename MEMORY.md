# ذاكرة المشروع — ProStudio

> آخر مراجعة موثّقة: **2026-08-29** — كل رقم في هذه المذكرة مُخرَج من أمر `git`/`gh`/`pytest`
> نُفِّذ فعلاً في هذا التاريخ، وليس تقديراً.

## قاعدة العمل الدائمة

`arena/01a03969-prostudio` هو **الفرع الدائم للمستخدم** وهو **مصدر GitHub Pages**
(المسار `/docs`). أي تعديل يجب أن يصل إليه حتى يظهر على الموقع.

**لكن**: جلسة أرينا الحالية **مربوطة بمنصة التشغيل** على فرع الجلسة
`arena/01a04ec9-prostudio`، ولا يمكن للوكيل تحويل الجلسة إلى فرع آخر — هذا قرار
المنصة وليس اختياراً. لذلك سير العمل الإلزامي هو:

1. العمل على فرع الجلسة `arena/01a04ec9-prostudio`.
2. `git push origin arena/01a04ec9-prostudio`.
3. فتح PR منه إلى `arena/01a03969-prostudio` ودمجه.
4. GitHub Pages يعيد البناء تلقائياً من `arena/01a03969-prostudio:/docs`.

قبل أي تعديل: `git fetch` ثم `git merge --ff-only origin/arena/01a03969-prostudio`
للتأكد أن العمل يبدأ من آخر ما دفعه المستخدم، لا من نسخة قديمة.

## الهوية والروابط (مؤكّدة بـ `gh api` بتاريخ 2026-08-29)

| الشيء | القيمة |
|---|---|
| المستودع | https://github.com/dhiyaddineb-hue/prostudio |
| فرع المستخدم | https://github.com/dhiyaddineb-hue/prostudio/tree/arena/01a03969-prostudio |
| **الصفحة المنشورة** | **https://dhiyaddineb-hue.github.io/prostudio/** |
| صفحة إرسال ملف | https://dhiyaddineb-hue.github.io/prostudio/send.html |
| معاينة مقطع | https://dhiyaddineb-hue.github.io/prostudio/preview.html?p=<اسم-المشروع> |
| مصمم الأصوات | https://dhiyaddineb-hue.github.io/prostudio/studio.html |
| إعداد Pages | الفرع `arena/01a03969-prostudio`، المسار `/docs`، `build_type: legacy`، الحالة `built` |
| آخر بناء Pages | من الكوميت `ea677a7` في `2026-08-29T18:32:34Z` — ناجح |

## ما تم فهمه من المراجعة

ProStudio استوديو عربي لدبلجة وترجمة فيديوهات YouTube والملفات المحلية. المسار
الأساسي: التنزيل أو الرفع ← التفريغ ← الترجمة ← توليد الصوت ← مزامنة الصوت ومزج
الخلفية ← تصدير MP4 وSRT.

- **النواة** Python داخل `youtube_auto_dub/` — `core.py::run()` هو الخط الأنبوبي.
- **الواجهة المحلية** FastAPI داخل `web/app.py` على المنفذ 8080 (`0.0.0.0` افتراضياً،
  مع `proxy_headers=True`) — تُشغَّل بـ `python run_studio.py`.
- **واجهة GitHub Pages** ثابتة داخل `docs/` — تعرض الدبلجات الجاهزة، وترفع الملفات
  مباشرة عبر GitHub API (`github-upload.js`)، وتضم مصمم الأصوات.

### أرقام مرجعية (مُخرَجة من `git ls-files` بتاريخ 2026-08-29)

| المكوّن | العدد |
|---|---|
| الملفات المتتبَّعة في Git | **216** |
| وحدات Python في `youtube_auto_dub/` | **30** (+ `language_map.json` = 31 ملفاً) |
| ملفات `web/` | **11** |
| سكربتات `scripts/` | **13** |
| ملفات `tests/` | **31** (26 وحدة اختبار Python + `__init__.py` + 4 ملفات `.mjs` للمتصفح) |
| مشاريع الدبلجة | **4**: Bob-Proctor-5min-2v، Bob-Proctor-Sample، Phantom-Thread، Vikings-Ragnar-Floki |

### نتيجة الاختبارات (نُفِّذت فعلاً بتاريخ 2026-08-29)

```
.venv/bin/python -m pytest -q     →  171 passed, 3 skipped, 0 failed
node tests/browser/*.mjs          →  32 + 12 + 26 + 36 = 106 تأكيداً، 0 فشل
```

ملاحظة: اختبارات `.mjs` **ليست** حزم `node:test` — تُشغَّل مباشرة
(`node tests/browser/shrink.test.mjs`)؛ `node --test tests/browser/` يفشل.

## حدود طلب اللعبة الحالي

الرسائل الحالية تطلب فهم المشروع وتثبيت الفرع والروابط في الذاكرة، ولا تقدم تصميم
لعبة أو آليات لعب أو هدفاً بصرياً. لذلك لا ينبغي إنشاء لعبة أو تحويل المشروع إلى
Babylon/WebDev دون وصف إضافي صريح؛ عند وصول وصف لعبة لاحقاً، يجب أولاً قراءة مهارة
تطوير الألعاب ثم بناءها في نطاق منفصل دون كسر خط الدبلجة الحالي.
