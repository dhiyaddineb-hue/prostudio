# Lip-sync comparison

`run.py` is the safe production gate. It samples faces, fits the final dubbed audio to the video duration, runs Wav2Lip only when a face is visible, and remuxes the fitted dubbed audio so the original audio is not copied.

`compare.py` runs Wav2Lip and Diff2Lip on the same face-gated video and the same fitted audio, then writes `comparison.json`. Diff2Lip is invoked with an explicit command template because its research repository exposes `scripts/inference_single_video.sh` and `generate_dist.py` rather than a stable package CLI.

Example:

```bash
python lip_sync/compare.py \
  --video input/video.mp4 \
  --audio output/dubbed.wav \
  --wav2lip-repo /opt/Wav2Lip \
  --wav2lip-checkpoint /models/wav2lip.pth \
  --diff2lip-repo /opt/diff2lip \
  --diff2lip-checkpoint /models/diff2lip.pt \
  --diff2lip-command 'python {repo}/generate_dist.py --video_path {video} --audio_path {audio} --out_path {output} --model_path {checkpoint}' \
  --output-dir output/lipsync-compare
```

The Diff2Lip command must be verified against the checked-out revision and its required flags. The wrapper does not silently guess a command or claim a successful result if the expected output file is absent.

The audio pipeline before this directory is:

```text
source video → optional Demucs speech/background stems → pyannote speaker turns →
per-speaker VoxCPM/Seed-VC → per-segment timing fit → lip-sync backend → remux
```

`pyannote` is enabled with `--diarize` and `HF_TOKEN`; Demucs is enabled in the workflow with `separate_sources: true`. Both features have conservative fallbacks when their model, token, or output quality is unavailable.
