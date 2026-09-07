"""Regression tests for VoxCPM failure propagation."""

import asyncio

import pytest

import youtube_auto_dub.voxcpm_tts as voxcpm


def test_queue_full_reason_reaches_the_pipeline(monkeypatch, tmp_path):
    def queue_full(*_args, **_kwargs):
        raise RuntimeError("Queue is full! Please try again.")

    monkeypatch.setattr(voxcpm, "_BACKEND", "space")
    monkeypatch.setattr(voxcpm, "_generate_space", queue_full)
    monkeypatch.setenv("YAD_VOXCPM_ATTEMPTS", "1")

    with pytest.raises(RuntimeError, match="Queue is full"):
        asyncio.run(voxcpm.speak_voxcpm("اختبار", tmp_path / "out.wav", language="ar"))
