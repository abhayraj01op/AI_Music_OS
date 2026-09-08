# AI Music OS — FIXED

Google Colab notebooks for:
1. **ACE-Step** — real upstream repository; no fake sine-wave generator.
2. **Kokoro TTS** — persistent Hugging Face cache and Drive outputs.
3. **Voice Profiles** — persistent reference audio storage without falsely claiming MMS-TTS is voice cloning.

## Important

The previous package had two critical functional problems:
- ACE-Step was replaced by a synthetic sine-wave `launch.py`.
- The Qwen voice-clone notebook used `facebook/mms-tts-eng`, which is not a reference-voice cloning implementation.

Those misleading fallbacks have been removed.

## Google Drive structure

`MyDrive/AI_Music_OS/`

- `models/ace_step/`
- `cache/huggingface/`
- `outputs/ace_step/`
- `outputs/kokoro/`
- `outputs/qwen_voice/`
- `voice_bank/`
- `repos/ACE-Step/`

## Usage

Run each notebook in a separate Colab runtime.

### ACE-Step
The notebook clones/updates the upstream repository and installs its own dependency metadata. It never overwrites the repository's launcher. The exact entrypoint can change between ACE-Step releases, so the notebook detects common upstream entrypoints rather than inventing one.

### Kokoro
The notebook detects the Colab Python version. For Python 3.13+, it installs the current upstream Kokoro/Misaki repositories because the PyPI Kokoro 0.9.4 release declares Python `<3.13`. It installs `espeak-ng`, runs a generation smoke test, then starts Gradio. Outputs are saved to Drive. No Cloudflare tunnel is required.

### Voice Profiles
The notebook saves reference recordings to Drive. It intentionally refuses to generate cloned speech until a genuine speaker-conditioned backend is installed. This is preferable to returning fake audio or mislabeled TTS output.

## Reproducibility

AI packages change frequently. For maximum reproducibility, pin the exact package versions after testing the chosen Colab runtime/GPU combination.

## Status

This package fixes the misleading/fake implementations identified in the original ZIP. It does not claim that a generic TTS model is a voice-cloning model.
