# AI Music OS v6

Three real, free model notebooks: ACE-Step music generation, Kokoro speech, and F5-TTS reference-voice cloning. The legacy `03_Qwen_Voice_Clone_Colab.ipynb` filename is retained for existing GitHub links; its actual model is F5-TTS, not Qwen.

## Setup correction
The previous setup tried creating a virtual environment with Colab's system Python; its ensurepip failed before any AI dependency installed. V6 extracts the official pinned uv binary from its wheel, then uses uv to create the managed Python environment. It does not run system venv/ensurepip or install uv into the Colab kernel.

## Run
1. Replace **all three** files inside your GitHub repository's `colab/` folder, or upload one notebook directly to Colab.
2. Disconnect and delete the old Colab runtime. Choose T4 GPU and Run all.
3. Allow Google Drive, wait for installation/model download, then open the printed Gradio URL.
4. Confirm the notebook heading says ACE-Step V6, Kokoro V8 or F5-TTS V6; the ACE screenshot showed the old requirements.txt installer, not the isolated version.
5. Use one notebook per session. Keep Colab open while using its interface.

Each notebook is self-contained: pinned requirements and app code are embedded. The extra Python apps and requirements are included for inspection and testing, not required uploads. Dependencies use an isolated Python 3.11.18 environment with matching PyTorch 2.6 / CUDA 12.4 audio and vision packages. Colab's preinstalled packages are not changed. No restart or Python kernel switching is required.

## Drive
Models/cache: `MyDrive/AI_Music_OS/cache/huggingface` and `models/ace_step`. WAVs and logs: `outputs/ace_step`, `outputs/kokoro`, `outputs/qwen_voice`. Voice profiles: `voice_bank`. Legacy saved voice profiles remain usable. Existing output folders are preserved.

## Limits
Installation needs internet and temporary disk (allow 10–15 GB plus model space); first model downloads can take several minutes. Free Drive is 15 GB total: all three models plus other projects may exceed it. Model licences apply; clone only voices with permission. F5-TTS principally supports English/Chinese, not guaranteed 100+ languages. ACE-Step CPU offload reduces GPU demand but free T4 generation is still slow and may run out of memory for long songs. Start with short generations.

Full package resolution, imports, official CLI flags and CPU regressions were tested. GPU generation in an actual Colab T4 session remains unverified; no claim of guaranteed full operation is made. See FIX_REPORT.md for the exact checks and remaining limits.
