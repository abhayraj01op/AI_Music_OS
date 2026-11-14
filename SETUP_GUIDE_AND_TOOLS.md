# AI OS – Tools, Sizes & Setup Guide (v3, 2026-10-28)

All tools are free and open-source. Sizes are approximate download sizes (what lands in Google Drive).
Free Colab T4 = 15 GB GPU memory, ~78 GB temporary disk, Drive free = 15 GB.

## 1. Tools already used (download sizes)

| Project | Tool | Used for | Size |
|---|---|---|---|
| Creator | SDXL base 1.0 (fp16) | image base | ~6.9 GB |
| Creator | SDXL-Lightning 4-step UNet | fast images (~3-5 s on T4) | ~5.1 GB |
| Creator | SDXL-Turbo (fp16) | fastest preview images | ~6.9 GB |
| Creator | SDXL VAE fp16 fix | color fix | ~0.3 GB |
| Creator | Real-ESRGAN x4 | upscale | ~0.07 GB |
| Creator | Rembg (u2net) | remove background | ~0.17 GB |
| Creator | IP-Adapter FaceID + InsightFace | same face in images | ~1.5 GB |
| Creator | LTX-Video (2B, chunked to Drive) | text/image to video | ~9.5 GB |
| Creator | FFmpeg | video editing | built into Colab |
| Creator | Qwen3-4B-Instruct-2507 (NEW, replaced Qwen2.5-3B) | writer / agents | ~8 GB |
| Creator | Qwen3-1.7B (NEW, replaced Qwen2.5-1.5B) | light writer | ~3.4 GB |
| Creator | Phi-4-mini (NEW, replaced Phi-3.5-mini) | writer option | ~7.7 GB |
| Creator/Coding | Qwen2.5-Coder-3B / 7B-AWQ | coding | ~6 GB / ~5.5 GB |
| Creator/Media | faster-whisper large-v3-turbo | subtitles / speech-to-text | ~1.6 GB |
| Creator/Music | Demucs htdemucs | split vocals/music | ~0.3 GB |
| Creator/Voice | Kokoro-82M | fast natural TTS | ~0.33 GB |
| Music | ACE-Step v1 3.5B | songs with vocals | ~7.7 GB |
| Music/Voice | F5-TTS | voice cloning | ~1.4 GB |
| Voice | MMS-TTS | speech in 1100+ languages | ~0.15 GB per language |

Total if you keep everything: ~85 GB. Free Drive (15 GB) cannot hold it all — keep only what you use,
or let the notebooks download to Colab disk each session (deleted when the session ends).

## 2. Newer free tools recommended (2026)

| Replace / add | New tool | Why | Size | Fits free T4? |
|---|---|---|---|---|
| SDXL images | FLUX.1-schnell (GGUF Q4/Q8) | far more realistic, 4 steps | ~7–12 GB | Yes (GGUF) |
| SDXL images | Z-Image-Turbo (already referenced) | photoreal, 8 steps, fast | ~12 GB | Yes, tight |
| LTX-Video 2B | Wan 2.2 TI2V-5B | best free realistic video | ~10 GB | Yes, slow (480p) |
| LTX-Video 2B | LTX-Video 0.9.8 distilled | fastest video | ~6 GB | Yes |
| Voice clone | Chatterbox Multilingual | clone in 23 languages, emotion control | ~3 GB | Yes |
| Voice clone | XTTS-v2 | clone in 17 languages | ~1.9 GB | Yes |
| Voice clone 100+ langs | MMS-TTS + OpenVoice v2 tone converter | speak in 1100+ languages, then convert to cloned voice | ~0.5 GB + 0.15/lang | Yes |
| Coding | Qwen2.5-Coder-7B-AWQ (keep) | best coder that fits T4 | ~5.5 GB | Yes |
| Speech-to-text | faster-whisper large-v3-turbo (keep) | still the best free | ~1.6 GB | Yes |

Honest limits: no single free model clones a voice in 100+ languages; the MMS + OpenVoice combo is the
closest free route. Image/video models will still refuse nothing technically, but you are responsible
for following the law and the model licences (no real-person deepfakes without consent).

## 3. Upload to GitHub (once per project)

1. Go to github.com → **New repository** → name it exactly like the folder (e.g. `AI_Creator_OS`) → Public → Create.
2. On the empty repo page click **uploading an existing file**.
3. Unzip the project zip on your computer, drag **all files inside the folder** (not the folder itself) into the page.
4. Click **Commit changes**.

## 4. Open in Colab

Use this link pattern (replace USERNAME and the notebook path):

`https://colab.research.google.com/github/USERNAME/AI_Creator_OS/blob/main/notebooks/00_AI_CREATOR_OS_MASTER.ipynb`

| Project | First notebook to open |
|---|---|
| AI_Creator_OS | `notebooks/00_AI_CREATOR_OS_MASTER.ipynb`, then 01–08 |
| AI_Media_OS | `colab/AI_Media_OS.ipynb`|
| AI_Music_OS | `colab/01_ACE_Step_Colab.ipynb` (also 02, 03)|
| AI_Coding_OS | `colab/AI_Coding_OS.ipynb`|
| AI_VoiceClone_OS | `colab/AI_VoiceClone_OS.ipynb`|

Then in Colab: **Runtime → Change runtime type → T4 GPU → Save**, then **Runtime → Run all**.
Allow Google Drive when asked — models are saved there so the next run is faster.

Shortcut without GitHub: open colab.research.google.com → **Upload** → pick the .ipynb from the unzipped folder
(the notebooks that clone code from GitHub still need step 3).
