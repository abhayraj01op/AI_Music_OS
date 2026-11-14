# Fix report

## Second-pass fixes

- Fixed the Kokoro Colab installation failure on Python 3.13.
- The old notebook requested `kokoro>=0.9.4` from PyPI, but that release declares Python `<3.13`; current Colab can use Python 3.13.
- On Python 3.13+, the notebook now installs the current upstream `hexgrad/kokoro` and `hexgrad/misaki` repositories.
- Added `espeak-ng` installation before Kokoro initialization.
- Removed unnecessary forced Gradio package upgrades from the Kokoro install step.
- Fixed Kokoro generator handling: current usage returns `(graphemes, phonemes, audio)`, not a dictionary.
- Added a real generation smoke test before starting Gradio.
- Drive mount is safe when Drive is already mounted.
- Outputs remain persistent under `MyDrive/AI_Music_OS/outputs/kokoro`.

The original ACE-Step and fake voice-cloning fixes remain unchanged.

## v4 — Kokoro missing `addict` (2026-11-14)

- Screenshot: Python 3.13 fails while importing `misaki.token`, because `addict` is missing.
- Cause: the previous installer used `--no-deps` for Kokoro/Misaki but only installed four dependencies manually. Misaki's base and English dependencies were skipped.
- Replaced that partial installation with dependency-resolved, pinned official Kokoro and `misaki[en]` sources.
- Preserved installed NumPy/SciPy/PyTorch/audio/vision versions using pip constraints. If incompatible, installation fails visibly rather than silently replacing the GPU stack.
- Included SoundFile and Gradio in the install, and added import checks before model loading.
- ACE-Step and F5-TTS notebooks are unchanged. No paid APIs added.
- Local checks cover the missing dependency and installer behavior; full speech generation on a Colab T4 remains unverified.

### Use the correction
1. Unzip this package and replace `colab/02_Kokoro_Colab.ipynb` in your GitHub repository.
2. Open the updated notebook in Colab (or upload the notebook directly).
3. Choose Runtime → Disconnect and delete runtime, reconnect with T4, then Run all. Allow Google Drive access.
4. Step 4 should print `addict + Misaki English phonemizer: OK`; Step 7 generates the test WAV.

### Immediate repair for the existing session
Run a new cell before Step 5:

```python
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', 'addict'], check=True)
from misaki.token import MToken
print('addict: OK')
```

This repairs only the shown missing module. Use the corrected installer to resolve the rest of the dependency chain.

## v5 — All three notebooks checked (2026-11-14)

### What the screenshot establishes
The install command failed, but the screenshot does not show pip's preceding explanation, so its exact failing dependency cannot be identified from that image alone. It does show Python 3.13 with NumPy 2.4.6, torch 2.14.0+cu130, torchaudio 2.11.0+cu130 and torchvision 0.29.0+cu130. These are not a matching tested audio/vision stack. v4's blanket core-version constraints did not repair that environment. The cited web error `src/main.tsx ... isExpanded` is unrelated to the notebook; that file is absent from this TanStack project and the Music archive. No unrelated web functionality was changed.

### Repairs
- Replaced all three installation paths with separate Python 3.11.18 environments, matching torch/torchaudio 2.6.0+cu124 and torchvision 0.21.0+cu124. The Colab kernel's packages are untouched.
- Included complete resolved requirements inside each notebook and as readable locks. Installs fail with the last 18,000 characters of the actual log, with complete logs on Drive. No `--no-deps` workaround.
- Kokoro: full English extras, addict, pip and spaCy's English model; real audio tensor conversion before NumPy/WAV writing; correct American/British phonemizers with a shared model; unique output names; real speech smoke test before interface launch.
- ACE-Step: pinned official upstream source; tested CLI options; CPU offload, no bf16, no compile, overlapped decode. Working directory is on Drive so upstream `./outputs` persists. An actual dependency resolution exposed `py3langid==0.3.0` requiring NumPy >=2: ACE uses NumPy 2.2.6, while Kokoro/F5 use 1.26.4 in their separate environments.
- F5-TTS: pinned 1.1.10 and compatible x-transformers/torchcodec; pinned PyArrow 19.0.1 after a real import test found newer PyArrow requiring NumPy 2 despite the resolver accepting NumPy 1.26.4. Reference WAVs/transcripts remain persistent; profile validation prevents selecting files outside the saved voice bank. Filenames are unique.
- Existing voice-bank and output locations preserved. Fixed stale README language that described a profile-only feature rather than the existing F5-TTS implementation. Kept legacy notebook filenames for GitHub links.
- Startup checks test the isolated interpreter's CUDA rather than the notebook kernel's torch. Repeated launch terminates the prior notebook process. Timeout/exit shows actual logs instead of printing a fake success URL.

### Checks actually executed
- Resolved all three complete CUDA 12.4 requirement locks for Python 3.11.
- Installed each full requirement set into a separate clean CPU environment (GPU wheel pins replaced by matching CPU wheel pins for local testing). Package consistency check and Kokoro/F5/ACE import checks passed in each environment.
- Official ACE-Step `--help` succeeded and all notebook options exist.
- Kokoro model downloaded and generated actual speech on CPU: 3.675 seconds, 24 kHz WAV, finite/nonzero audio; file readback passed. This was not a simulated model response.
- Eight regression checks passed: all notebook cells compile, isolated install behavior, complete phonemizer requirements, official T4 flags, tensor conversion/British voice WAV handling, empty input validation, real FFmpeg profile conversion with mocked F5 inference, and invalid profile rejection.
- F5 inference in the regression is mocked; ACE/F5 model synthesis and any Gradio share-link generation are NOT claimed as tested.

### Remaining verification
No Colab T4 GPU is available here. CUDA driver compatibility, real ACE-Step song generation, real F5 voice synthesis and hosted Gradio interactions must still be checked in Colab. Local resolution/import/CPU tests do not certify all three as fully working on every future Colab runtime. Free Colab can disconnect and long music generations can exceed T4 memory; start short.

### Run this version
Replace **all three** matching notebook files in GitHub, or upload the chosen notebook directly to Colab. Disconnect and delete the old runtime, select T4 GPU, then Run all. Open the printed Gradio link and keep the session open. Notebook titles should show Kokoro V7 / ACE-Step V5 / F5-TTS V5; do not rely on a stale browser-tab title from an older notebook.

### Local regression command
Use Python 3.11 with the Kokoro dependencies installed, plus FFmpeg available:
`python -m unittest discover -s tests -v`

## v6 — Colab ensurepip setup failure (2026-11-14)

### Diagnosis
The voice-clone screenshot identifies the actual failing command: creating the bootstrap environment through `/usr/bin/python3 -m venv` invokes ensurepip, which exits with status 1. No F5-TTS model code had run yet. Its underlying ensurepip output is not included, so we do not invent a more specific OS cause. This bootstrap was shared by all three v5 notebooks and is corrected in all three.

The ACE-Step screenshot is an older notebook: it still clones a Drive repository and installs its requirements directly with system Python 3.13. It is not the v5 isolated ACE notebook. The preceding pip explanation is not shown, so that exact dependency failure cannot be determined from the image. Use the replacement ACE-Step V6 file, not the older saved Colab copy.

The cited `src/App.tsx` event-handler error does not match these Colab failures; that app file is absent. No unrelated web source was changed.

### Changes
- All three notebooks download the official pinned uv 0.13.8 wheel using the kernel's existing pip, extract only its Linux binary, and create their managed Python environments through uv. No system venv/ensurepip, no shell installer, and no uv installation into the Colab kernel.
- Wheel download uses PyPI, binary-only/no-dependency download. The only `--no-deps` here is for downloading the self-contained uv binary; AI dependencies still use full locked synchronization and consistency checks.
- Atomic executable replacement, executable permissions, and version check; successful reruns reuse the shared binary and per-tool environment. Failed downloads show real logs and stop execution.
- New `/content/ai_music_v6` temporary workspace avoids the failed v5 bootstrap. Drive models, voice profiles and outputs are unchanged. Install/launch logs are versioned v6.
- Remove inherited PYTHONPATH/PYTHONHOME and disable user-site packages for child processes so Colab kernel paths do not leak into isolated environments.
- Requirement locks and model application sources are unchanged from their previously tested versions; no features removed.

### Verification actually completed
- Executed the exact first setup cell of each new notebook from sandbox Python 3.13, with Drive mounting replaced by a temporary test folder. Real PyPI wheel download, binary extraction, uv Python management and per-tool environment creation succeeded for ACE, Kokoro and F5; a child interpreter confirmed Python 3.11.18 in each.
- Executed each cell again; successful reruns did not redownload the binary or recreate existing tool environments. A guard rejected any attempted system venv/ensurepip call throughout testing.
- Ten CPU regression tests passed, including new bootstrap/rerun behavior and failure-log handling checks.
- Rechecked official ACE CLI help and F5 API import; reran real Kokoro CPU inference with 3.675-second WAV readback.
- Full CUDA locks and model applications remain byte-identical to v5. Actual Colab network/OS/Drive integration, T4 song generation, F5 reference-conditioned synthesis and live Gradio links remain unverified. These local checks are not a guarantee of complete Colab execution.

### Required replacement
1. Unzip v6 and replace all three matching files in your GitHub `colab/` folder. Alternatively upload a notebook directly to Colab to avoid opening an older GitHub copy.
2. Close the old Colab notebook; open the replaced file. Check the first visible heading: ACE-Step V6 / Kokoro V8 / F5-TTS V6. The first setup cell must mention wheel extraction rather than system venv.
3. Disconnect and delete the old runtime; select T4 GPU, then Run all and allow Drive.
4. Open the printed Gradio URL. Keep Colab connected. Logs are on Drive beside outputs.


## v7 — Expanded TTS and voice-cloning workflows (2026-11-14)
- Added 41 official Kokoro voices in seven language/accent groups; all original voices retained. Catalogue verified against `hexgrad/Kokoro-82M` upstream voice files. Existing locks contain the Espeak phonemizer needed by the five added non-English groups. No extra AI dependency or paid API.
- Added lazy shared-model language pipelines, bounded text splitting, speed/pauses, weighted same-language blending, pitch/loudness postprocessing, three audio formats, text import, batch ZIP, chunk-level approximate SRT and output history.
- F5 now includes reference trimming/normalization, recording, safe overwrite, preview/transcript editing, official API sampling/seed/guidance/crossfade/silence controls, batch export and multi-profile dialogue. English/Chinese limitations remain explicit; legacy Qwen filename retained without claiming a Qwen model.
- Added `audio_tools.py` and `tests/test_features.py`; changed both speech app sources, both speech notebook embedded apps, existing regression fixture and README/report. ACE notebook and all dependency locks remain byte-identical to v6. uv wheel bootstrap preserved; helper code and output environment are embedded in both notebooks.
- 23 regressions passed, including actual FFmpeg audio conversions, profile trims and readback; model forwarding/dialogue uses mock F5. Real Kokoro CPU synthesis passed all seven language/accent groups plus weighted blending; nonzero finite 24 kHz audio read back successfully. Not every voice was synthesized.
- The reported `src/lib/utils.ts(5,11)` unknown-object error was not reproduced: current helper is `twMerge(clsx(inputs))`, with typed imports and no unknown dereference. The existing web routing test passed; no unrelated website changes made.
- Remaining: actual Colab T4 generation, real F5/ACE inference and hosted Gradio links not tested. Japanese/Chinese Kokoro extras not included, to avoid changing the proven dependency locks.

- Local live Kokoro interface generated a playable 3.65-second WAV with audio/subtitle download links. Desktop screenshot inspected; phone had no horizontal overflow. Added minimum control widths to avoid clipping long language names on narrow screens. F5 interface construction passed, but its live synthesis flow was not exercised.

## v8 — full ACE-Step and genuine Qwen upgrade (2026-11-14)
request3385.txt requests full ACE-Step and Qwen voice-clone upgrades, not a website feature. Notebook 01 now installs official ACE-Step 1.5 pinned ca1e85fe9430179831e6bc6be790c332190a3866 and its full upstream frozen lock; Python 3.12 avoids FlashAttention's 3.11-only marker. Full 2B/XL and planner options remain available with an unrestricted official UI; T4 defaults use offload/PT/batch one. Notebook 03 now loads genuine Qwen3-TTS pinned source 022e286b98fbec7e1e916cb940cdf532cd9f488e, full 1.7B Base/CustomVoice/VoiceDesign or optional 0.6B, with float16/SDPA and the official clone/save-load/preset/design screens. Existing F5 and older ACE preserved in notebooks 04/05, Kokoro unchanged. Embedded helpers/bootstrap and existing Drive paths retained. Added model_config.py, qwen_app.py, two locks and eight upgrade regressions; README and this report updated.

31 final checks passed. Before sandbox memory loss, full ACE frozen install/consistency/CLI help and genuine Qwen import succeeded. Rebuilt the final package using the smaller, faithful official Qwen UI rather than the experimental custom UI; final Qwen CPU install/consistency/imports and desktop/phone UI inspection succeeded. UI inspection used a non-inference stub. The genuine CPU Qwen synthesis attempt exhausted local memory; no real output or T4 certification is claimed. Real Colab/Drive/T4 synthesis, larger full models, training and public share links remain unverified. Reported src/main.tsx is absent from this TanStack project; website routing test passed and no unrelated website code was modified.
