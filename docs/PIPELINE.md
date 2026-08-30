# ProStudio dubbing pipeline

`pipeline.py` is the single orchestration entry point. It accepts an approved translated/TTS track and coordinates optional source separation, conservative diarization, Seed-VC enhancement, timing fit, conditional Wav2Lip, remuxing, and a JSON quality report.

```bash
python pipeline.py --video input.mp4 --dubbed-audio translated.wav \
  --output output/dub.mp4 --profile config/pipeline_profile.yaml --mode safe
```

Use `--dry-run` to validate paths, profile resolution, and token visibility without loading expensive models. Override individual values with repeatable `--set key=value`, for example `--set languages.target=fr --set timing.max_tempo_factor=1.1`.

The supplied modes are `safe`, `high_quality_single`, `multi_speaker_cinematic`, and `experiment`. Safe mode disables all optional model stages and never preserves background audio. The cinematic mode enables Demucs, pyannote, separated background preservation, and lip-sync, but each remains conditional on available credentials, checkpoints, and quality gates.

Diarization only labels a transcript segment when a real pyannote turn covers at least the configured overlap threshold (0.55 by default). It does not fabricate a whole-video speaker. Demucs outputs are checked for existence, duration, and usable metadata; contamination remains reported as **unknown** because Demucs is not a cinema-dialogue isolation guarantee. If either optional stage fails, the pipeline falls back to generated dialogue audio rather than remixing the original mixed track.

Timing is currently guaranteed at the final track boundary by the existing `fit_audio` adapter. Segment-aware TTS adapters can provide per-segment audio and timing manifests; the report explicitly marks the current whole-track fallback as `per_segment: false` rather than claiming otherwise. Lip-sync is Wav2Lip-only in production mode, is face-gated, and is reported as `unvalidated` unless an independent synchrony metric is available. Diff2Lip remains an experiment adapter and must never replace the approved output automatically.

Every successful run writes `<output>.pipeline.json`. It records the resolved configuration, stage requests and outcomes, fallback events, model endpoint, timing data, output stream checks, duration comparison, and warnings. Intermediate references and model artifacts should be kept outside published artifacts unless a debug policy explicitly requests them.

The open-source Wav2Lip implementation has upstream research/academic/personal-use restrictions. Confirm licensing before any commercial deployment.
