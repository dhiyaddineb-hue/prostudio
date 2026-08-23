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

## المتطلبات

- Python 3.10+
- FFmpeg (أو الحزمة `imageio-ffmpeg` المضمّنة)
- اتصال إنترنت للترجمة وEdge-TTS وتحميل يوتيوب

## الترخيص

MIT. نواة الخط الأنبوبي من youtube-auto-dub © Nguyen Cong Thuan Huy.
