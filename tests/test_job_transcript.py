"""A user's own video must never be dubbed with the bundled demo narration."""

import web.app as webapp


def test_demo_script_only_applies_to_demo_source():
    """The fallback narration is tied to the demo clip, not to any upload."""
    src = webapp.inspect_transcript(
        transcript="", source="/uploads/user-clip.mp4", whisper=True
    )
    assert src == ""

    demo = webapp.inspect_transcript(
        transcript="", source=str(webapp.DEMO_SOURCE), whisper=True
    )
    assert demo == webapp.DEMO_SCRIPT


def test_user_transcript_wins():
    text = "هذا نص المستخدم."
    assert (
        webapp.inspect_transcript(
            transcript=text, source=str(webapp.DEMO_SOURCE), whisper=True
        )
        == text
    )


def test_missing_transcript_without_whisper_is_rejected():
    """Without Whisper and without a transcript there is nothing to dub."""
    try:
        webapp.inspect_transcript(
            transcript="", source="/uploads/user-clip.mp4", whisper=False
        )
    except RuntimeError as exc:
        assert "Whisper" in str(exc) or "التفريغ" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
