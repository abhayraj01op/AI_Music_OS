"""Full official Qwen3-TTS demo with T4-compatible loading and Drive exports."""
import os
from pathlib import Path
from model_config import qwen_settings


def main():
    import torch
    from qwen_tts import Qwen3TTSModel
    from qwen_tts.cli.demo import build_demo
    model_id = os.environ['AI_MUSIC_QWEN_MODEL']
    settings = qwen_settings(model_id)
    settings['dtype'] = torch.float16
    out = Path(os.environ['AI_MUSIC_OUT'])
    out.mkdir(parents=True, exist_ok=True)
    print('Loading full official model:', model_id, flush=True)
    model = Qwen3TTSModel.from_pretrained(model_id, **settings)
    demo = build_demo(model, model_id, {'max_new_tokens': 2048})
    demo.queue(default_concurrency_limit=1).launch(share=True, server_name='0.0.0.0', server_port=7867, allowed_paths=[str(out)])


if __name__ == '__main__':
    main()
