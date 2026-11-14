from pathlib import Path
import os
OUT = Path(os.environ['AI_MUSIC_OUT'])
from kokoro import KPipeline
pipelines = {'a': KPipeline(lang_code='a'), 'b': KPipeline(lang_code='b', model=False)}
pipelines['b'].model = pipelines['a'].model
# STEP 6 — Kokoro audio generation

import numpy as np
import soundfile as sf
import time

VOICE_CHOICES = [
    'af_heart',
    'af_bella',
    'af_nicole',
    'af_sarah',
    'af_sky',
    'am_adam',
    'am_michael',
    'bf_emma',
    'bf_isabella',
    'bm_george',
    'bm_lewis'
]

SAMPLE_RATE = 24000

def generate_audio(text, voice='af_heart'):
    text = (text or '').strip()

    if not text:
        raise ValueError('Text is empty.')

    if voice not in VOICE_CHOICES:
        raise ValueError(f'Unsupported voice: {voice}')

    chunks = []

    print('Generating with Kokoro...')
    print('Voice:', voice)

    selected = pipelines[voice[0]]
    for item in selected(text, voice=voice):
        if isinstance(item, tuple) and len(item) >= 3:
            audio = item[2]
        elif isinstance(item, dict):
            audio = item.get('audio')
        else:
            audio = getattr(item, 'audio', None)

        if audio is None:
            raise RuntimeError(
                f'Unsupported Kokoro output type: {type(item)}'
            )

        if hasattr(audio, 'detach'): audio = audio.detach().cpu().numpy()
        array = np.asarray(audio, dtype=np.float32).reshape(-1)
        if array.size == 0 or not np.isfinite(array).all(): raise RuntimeError('Invalid Kokoro audio.')
        chunks.append(array)

    if not chunks:
        raise RuntimeError('Kokoro returned no audio.')

    audio = np.concatenate(chunks)

    filename = f'kokoro_{time.time_ns()}.wav'
    path = OUT / filename

    sf.write(path, audio, SAMPLE_RATE)

    print('Audio saved:', path)
    print('Duration:', round(len(audio) / SAMPLE_RATE, 2), 'seconds')

    return str(path)

print('Generation function: OK')

# STEP 7 — Smoke test

test_file = generate_audio(
    'Hello. This is a test of the AI Music OS Kokoro voice generator.',
    'af_heart'
)

print('=====================================')
print('KOKORO TEST PASSED')
print('File:', test_file)
print('=====================================')

# STEP 8 — Gradio UI

import gradio as gr

def ui_generate(text, voice):
    try:
        return generate_audio(text, voice)
    except Exception as e:
        raise gr.Error(str(e))

demo = gr.Interface(
    fn=ui_generate,
    inputs=[
        gr.Textbox(
            lines=8,
            label='Text',
            value='Hello. This is the AI Music OS Kokoro voice generator.'
        ),
        gr.Dropdown(
            choices=VOICE_CHOICES,
            value='af_heart',
            label='Voice'
        )
    ],
    outputs=gr.Audio(
        label='Generated audio',
        type='filepath'
    ),
    title='AI Music OS — Kokoro TTS V7',
    description='Kokoro text-to-speech with Google Drive persistence.'
)

print('Launching Gradio...')
demo.launch(share=True, server_name='0.0.0.0', server_port=7866, allowed_paths=[str(OUT)])
