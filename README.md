# AI Music OS v8 — full official model upgrades

## Upload and run
Replace notebooks 01 and 03 in your GitHub `colab/` folder, or upload each directly to Colab. Keep notebook 02 (Kokoro) unchanged. Add notebook 04 for the preserved F5 tools and notebook 05 for legacy ACE-Step. Reset the runtime, select T4, allow Drive and Run all. Choose a model in cell 4 **before** Run all; changing it later requires rerunning cell 4 and launch. Use only one notebook/model per session.

| Notebook | Actual engine | Options |
|---|---|---|
| 01_ACE_Step_Colab.ipynb | ACE-Step 1.5 | Full 2B Turbo/SFT/Base; optional 4B XL Turbo/SFT/Base; 0.6B/1.7B/4B planner or none |
| 02_Kokoro_Colab.ipynb | Kokoro v7 unchanged | 41 voices, seven language/accent groups, blending, batch and exports |
| 03_Qwen_Voice_Clone_Colab.ipynb | **Genuine Qwen3-TTS** | Full 1.7B Base clone, CustomVoice and VoiceDesign; optional 0.6B Base/CustomVoice |
| 04_F5_Voice_Clone_Colab.ipynb | Preserved F5-TTS v7 | Existing saved profiles, trimming, batch ZIP and dialogue |
| 05_ACE_Legacy_Colab.ipynb | Preserved older ACE-Step | Previous tested installer/engine |

## ACE-Step full interface
Default: full 2B Turbo, 0.6B planner, CPU offload, PyTorch backend, no FlashAttention, batch one. Official interface is not restricted by service mode: generation, lyric/BPM/key/duration controls, reference/cover/repaint editing and upstream training tools are available. Base variants provide additional upstream task support; not every model supports every task. XL/4B planner/training may exceed free T4 RAM/VRAM. They are selectable full checkpoints, not a guarantee they fit. Start with short songs and use no planner if memory is tight.

ACE uses pinned official source and its full frozen uv.lock, including its bundled nano-vllm source. Python 3.12.16 avoids upstream's Python 3.11-only FlashAttention dependency. Torch/audio/vision are matched CUDA 12.8 versions from the official lock; current Colab driver compatibility remains unverified.

## Qwen full interface
The old Qwen filename previously ran F5. It now really loads Qwen3-TTS. Select one model:
- **1.7B Base:** reference audio + exact transcript, or embedding-only cloning without transcript; reusable save/load voice prompts in the official interface.
- **1.7B CustomVoice:** nine preset speakers with natural-language delivery instructions.
- **1.7B VoiceDesign:** describe a new voice/style in words.
- **0.6B Base/CustomVoice:** smaller alternatives. 0.6B CustomVoice does not offer the same instruction capability as 1.7B.

Supported languages: English, Chinese, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian. **Hindi is not supported by Qwen3-TTS; use Kokoro for Hindi.** It is not a 100-language model. Only clone voices you have permission to use. Do not load untrusted serialized saved-voice files.

T4 loading uses float16 and SDPA, not unsupported FlashAttention/bfloat16. Python 3.11.18, torch/audio 2.6 CUDA 12.4 and the complete pinned Qwen lock are isolated from the Colab kernel. SoX/FFmpeg are installed. Concurrency one; selected models download on demand, not all together. Model switching requires restarting the notebook application, not loading five models simultaneously.

## Preserved storage and setup
All notebooks remain standalone with embedded locks/application files. Bootstrap extracts pinned uv from an official wheel using existing pip; no system venv/ensurepip or Colab kernel package changes. Existing cache, voice_bank and output folders remain. ACE 1.5 checkpoints use `models/ace_step_15`; source-relative checkpoints and gradio_outputs link only to their exact Drive folders. Qwen generated files persist in `outputs/qwen_voice/qwen_generated`; use the audio player's download control. F5 profile/advanced workflows remain in notebook 04 rather than falsely claiming those custom features belong to Qwen.

“Full” means genuine full model checkpoints and official interfaces, **not model weights bundled in this ZIP**. Internet is required. Free Drive has 15 GB total; downloading all variants is impractical. Allow roughly 15 GB or more temporary installation space for ACE plus selected weights. XL weights alone are about 9 GB; other weights/cache add more. Download only what you need. Free Colab can disconnect and public Gradio links expire.

## Verification and limits
31 final regressions passed, including all preserved feature tests and eight upgrade tests. Complete official ACE installation, package consistency and actual CLI help passed locally before a later test exhausted sandbox memory. Qwen complete CPU-equivalent installation and genuine API imports passed; its official clone/save-load interface was inspected on desktop/phone (UI-only test model; no synthesized audio). A real Qwen CPU synthesis attempt exhausted local memory before a verified output. **Real Qwen/ACE generation, full 1.7B/XL inference, Colab T4, Drive integration, training and public share links remain unverified.** There is no guarantee of full T4 operation.
