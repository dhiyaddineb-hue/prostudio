"""Tests for the voice designer.

The properties that matter are the ones a wrong implementation breaks quietly:
neutral must be a true no-op, the pitch shifter must respect the file's own
sample rate, and a failed measurement must refuse rather than return a
plausible number.
"""

from __future__ import annotations

import math

import pytest

from youtube_auto_dub.voice_design import (
    PRESETS,
    VoiceSpec,
    clone_from_reference,
    ffmpeg_chain,
    parse_description,
)


def test_neutral_spec_is_a_no_op():
    assert VoiceSpec().is_neutral
    assert ffmpeg_chain(VoiceSpec(), 44100) == "anull"


def test_pitch_shift_uses_the_files_own_sample_rate():
    # Hard-coding 44100 here once halved every 24 kHz recording: asetrate
    # reinterprets the file at whatever rate it is given.
    at24 = ffmpeg_chain(VoiceSpec(pitch=12.0), 24000)
    at44 = ffmpeg_chain(VoiceSpec(pitch=12.0), 44100)
    assert "asetrate=48000" in at24 and "aresample=24000" in at24
    assert "asetrate=88200" in at44 and "aresample=44100" in at44


def test_pitch_shift_compensates_its_own_speed_change():
    # asetrate speeds the audio up as a side effect; without the inverse
    # atempo, "shift pitch" would silently also mean "speak faster".
    chain = ffmpeg_chain(VoiceSpec(pitch=12.0), 44100)
    assert "atempo=0.5" in chain


def test_rate_beyond_one_atempo_is_split_into_legal_steps():
    # ffmpeg rejects atempo outside 0.5-2.0, so a 0.6x request must not appear
    # as a single illegal filter.
    chain = ffmpeg_chain(VoiceSpec(rate=0.6), 44100)
    for step in [p for p in chain.split(",") if p.startswith("atempo=")]:
        assert 0.5 <= float(step.split("=")[1]) <= 2.0


def test_zero_sample_rate_is_refused():
    with pytest.raises(ValueError):
        ffmpeg_chain(VoiceSpec(pitch=1.0), 0)


def test_controls_are_clamped_to_audible_range():
    wild = VoiceSpec(pitch=99, rate=99, body=99, warmth=-99, clarity=99, air=-99)
    s = wild.clamped()
    assert s.pitch == 12.0 and s.rate == 1.6
    assert s.body == 8.0 and s.warmth == -8.0


def test_description_deep_male_lowers_pitch_and_adds_body():
    spec = parse_description("رجل عميق")
    assert spec.pitch < -3.0
    assert spec.body > 3.0


def test_description_child_raises_pitch_well_above_an_adult():
    assert parse_description("طفل").pitch > parse_description("شاب").pitch


def test_intensity_modifier_scales_the_effect():
    plain = parse_description("صوت عميق")
    strong = parse_description("صوت عميق جدا")
    assert abs(strong.pitch) > abs(plain.pitch)


def test_unknown_description_stays_neutral():
    # Inventing a transform for words we do not know would make every result
    # untrustworthy; leaving the base voice alone is the honest failure.
    assert parse_description("قطة تقود دراجة").is_neutral


def test_description_survives_diacritics_and_alef_variants():
    assert parse_description("رَجُل عَمِيق").pitch == parse_description("رجل عميق").pitch
    assert parse_description("إمرأة").pitch == parse_description("امراة").pitch


def test_clone_matches_reference_pitch_exactly():
    spec = clone_from_reference(
        ref_f0=200.0, ref_brightness=0.0, base_f0=100.0, base_brightness=0.0
    )
    assert spec.pitch == pytest.approx(12.0, abs=0.01)


def test_clone_refuses_a_silent_reference():
    # Every pitch detector returns 0 Hz on silence; a ratio against it is
    # meaningless, so this must raise rather than look confident.
    with pytest.raises(ValueError):
        clone_from_reference(0.0, 1.0, 120.0, 1.0)


def test_clone_rate_is_only_set_when_both_sides_were_measured():
    without = clone_from_reference(120.0, 1.0, 120.0, 1.0)
    assert without.rate == 1.0
    with_rate = clone_from_reference(120.0, 1.0, 120.0, 1.0, ref_rate=13.0, base_rate=10.0)
    assert with_rate.rate > 1.0


def test_spec_round_trips_through_a_dict():
    original = VoiceSpec(pitch=-3.5, rate=1.1, body=2.0, air=-1.0)
    assert VoiceSpec.from_dict(original.to_dict()).to_dict() == original.to_dict()


def test_from_dict_ignores_unknown_keys():
    assert VoiceSpec.from_dict({"pitch": 1.0, "nonsense": 5}).pitch == 1.0


def test_every_preset_actually_changes_something():
    for name, spec in PRESETS.items():
        assert not spec.is_neutral, name
        assert ffmpeg_chain(spec, 44100) != "anull", name


def test_atempo_alone_must_not_appear_to_transpose():
    """A time-stretch is not a transposition, and must not be verified as one.

    Measuring a stretched clip by median F0 gives a moving answer -- atempo=2.0
    reads 17% sharp -- purely because the clip is shorter, short voiced runs
    vanish, and the surviving frames are a different population. That artefact
    once looked like a broken pitch shifter. The chain is the thing under test,
    so assert on the chain: a pure rate change emits no resampling.
    """
    chain = ffmpeg_chain(VoiceSpec(rate=2.0), 44100)
    assert "asetrate" not in chain and "aresample" not in chain
    assert "atempo" in chain


def test_pitch_shift_emits_matched_resample_and_tempo_terms():
    """asetrate transposes and speeds up; the inverse atempo must undo exactly.

    Verified against audio separately: paired-frame measurement puts the error
    at 0.03-0.44 semitones across a two-octave range.
    """
    for semitones in (-12.0, -6.0, 6.0, 12.0):
        chain = ffmpeg_chain(VoiceSpec(pitch=semitones), 48000)
        ratio = 2.0 ** (semitones / 12.0)
        assert f"asetrate={int(round(48000 * ratio))}" in chain
        assert "aresample=48000" in chain
        product = 1.0
        for step in [p for p in chain.split(",") if p.startswith("atempo=")]:
            product *= float(step.split("=")[1])
        assert product == pytest.approx(1.0 / ratio, rel=1e-3)
