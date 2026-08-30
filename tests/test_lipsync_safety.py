import pytest
from lip_sync.run import atempo_chain


def test_atempo_chain_rejects_invalid_factor():
    with pytest.raises(ValueError):
        atempo_chain(0)


def test_atempo_chain_handles_ffmpeg_bounds():
    assert "atempo=2.0" in atempo_chain(4.0)
    assert "atempo=0.5000000" in atempo_chain(0.25)
