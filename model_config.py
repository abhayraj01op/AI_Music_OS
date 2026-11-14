"""Full official model options and conservative Colab launch settings."""
QWEN_MODELS = ['Qwen/Qwen3-TTS-12Hz-1.7B-Base', 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice', 'Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign', 'Qwen/Qwen3-TTS-12Hz-0.6B-Base', 'Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice']
ACE_MODELS = ['acestep-v15-turbo', 'acestep-v15-sft', 'acestep-v15-base', 'acestep-v15-xl-turbo', 'acestep-v15-xl-sft', 'acestep-v15-xl-base']
LM_MODELS = ['None', 'acestep-5Hz-lm-0.6B', 'acestep-5Hz-lm-1.7B', 'acestep-5Hz-lm-4B']


def ace_command(exe, model, lm, outputs):
    if model not in ACE_MODELS or lm not in LM_MODELS:
        raise ValueError('Unsupported ACE-Step model.')
    cmd = [str(exe), '--port', '7865', '--server-name', '0.0.0.0', '--share', '--service_mode', 'false', '--init_service', 'true', '--config_path', model, '--device', 'cuda', '--backend', 'pt', '--init_llm', 'false' if lm == 'None' else 'true', '--use_flash_attention', 'false', '--offload_to_cpu', 'true', '--offload_dit_to_cpu', 'true', '--quantization', 'int8_weight_only', '--batch_size', '1', '--allowed-path', str(outputs), '--download-source', 'huggingface']
    if lm != 'None': cmd += ['--lm_model_path', lm]
    return cmd


def qwen_settings(model_id):
    if model_id not in QWEN_MODELS:
        raise ValueError('Unsupported Qwen3-TTS model.')
    return dict(device_map='cuda:0', dtype='float16', attn_implementation='sdpa')
