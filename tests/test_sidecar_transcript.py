from pathlib import Path

def test_core_prefers_local_sidecar_srt_before_asr_cache():
    text=Path("youtube_auto_dub/core.py").read_text(encoding="utf-8")
    assert "Using trusted sidecar transcript" in text
    assert text.index("Using trusted sidecar transcript") < text.index("Using cached transcription")
