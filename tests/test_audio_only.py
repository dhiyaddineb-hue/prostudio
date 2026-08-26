"""Audio-only sources must dub without pretending to be video."""

import subprocess

import pytest

from youtube_auto_dub.audio import _has_video_stream, render_video
from youtube_auto_dub.ffmpeg_bin import ffmpeg_exe


def _make(path, args):
    subprocess.run(
        [ffmpeg_exe(), "-y", *args, str(path)],
        check=True, capture_output=True,
    )
    return path


@pytest.fixture
def tone(tmp_path):
    return _make(
        tmp_path / "tone.mp3",
        ["-f", "lavfi", "-i", "sine=frequency=220:duration=2", "-ac", "1", "-ar", "24000"],
    )


@pytest.fixture
def clip(tmp_path):
    return _make(
        tmp_path / "clip.mp4",
        ["-f", "lavfi", "-i", "color=c=black:s=160x120:d=2",
         "-f", "lavfi", "-i", "sine=frequency=220:duration=2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest"],
    )


def test_detects_audio_only(tone):
    assert _has_video_stream(tone) is False


def test_detects_real_video(clip):
    assert _has_video_stream(clip) is True


def test_render_audio_only_produces_playable_audio(tone, tmp_path):
    """The old code forced -map 0:v:0 and ffmpeg exited 234."""
    dub = _make(
        tmp_path / "dub.wav",
        ["-f", "lavfi", "-i", "sine=frequency=330:duration=2", "-ac", "1", "-ar", "24000"],
    )
    out = tmp_path / "out.mp3"
    render_video(video_path=tone, subtitle_path=None, dub_audio_path=dub, output_path=out)

    assert out.exists() and out.stat().st_size > 1000
    probe = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-i", str(out)],
        capture_output=True, text=True,
    ).stderr
    assert "Audio:" in probe


def test_render_video_still_muxes_video(clip, tmp_path):
    dub = _make(
        tmp_path / "dub.wav",
        ["-f", "lavfi", "-i", "sine=frequency=330:duration=2", "-ac", "1", "-ar", "24000"],
    )
    out = tmp_path / "out.mp4"
    render_video(video_path=clip, subtitle_path=None, dub_audio_path=dub, output_path=out)

    probe = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-i", str(out)],
        capture_output=True, text=True,
    ).stderr
    assert "Video:" in probe and "Audio:" in probe


def test_cover_art_is_not_mistaken_for_video(tmp_path):
    """An mp3 with embedded artwork is still audio-only."""
    art = _make(
        tmp_path / "art.png",
        ["-f", "lavfi", "-i", "color=c=red:s=64x64:d=1", "-frames:v", "1"],
    )
    tagged = tmp_path / "tagged.mp3"
    subprocess.run(
        [ffmpeg_exe(), "-y",
         "-f", "lavfi", "-i", "sine=frequency=220:duration=2",
         "-i", str(art), "-map", "0:a", "-map", "1:v",
         "-c:v", "mjpeg", "-disposition:v", "attached_pic",
         "-ac", "1", "-ar", "24000", str(tagged)],
        check=True, capture_output=True,
    )
    assert _has_video_stream(tagged) is False
