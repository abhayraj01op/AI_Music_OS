from pathlib import Path
import os
OUT = Path(os.environ['AI_MUSIC_OUT'])
VOICE_BANK = Path(os.environ['AI_MUSIC_VOICE_BANK'])
import gradio as gr
import soundfile as sf
from pathlib import Path
import shutil, time, subprocess

_tts = None
def get_tts():
    global _tts
    if _tts is None:
        from f5_tts.api import F5TTS
        print('⏳ Loading F5-TTS (first run downloads ~1.3 GB to Drive)...')
        _tts = F5TTS()
    return _tts

def profiles():
    return sorted(p.stem for p in VOICE_BANK.glob('*.wav'))

def save_voice(sample, profile_name, ref_text):
    if sample is None:
        raise gr.Error('Upload a voice sample first (5-15 seconds of clear speech works best).')
    safe = ''.join(c for c in (profile_name or '') if c.isalnum() or c in (' ', '_', '-')).strip()
    if not safe:
        raise gr.Error('Enter a profile name.')
    dst = VOICE_BANK / f'{safe}.wav'
    # Convert to mono 24 kHz WAV and trim to 15 s (F5-TTS works best with short references)
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(sample), '-ac', '1', '-ar', '24000', '-t', '15', str(dst)], check=True)
    (VOICE_BANK / f'{safe}.txt').write_text((ref_text or '').strip())
    return f'Saved: {dst}', gr.update(choices=profiles(), value=safe)

def generate(text, profile_name, speed):
    if not (text or '').strip():
        raise gr.Error('Enter text.')
    if not profile_name or profile_name not in profiles():
        raise gr.Error('Choose a saved voice profile.')
    ref = VOICE_BANK / f'{profile_name}.wav'
    if not profile_name or not ref.exists():
        raise gr.Error('Choose a saved voice profile.')
    ref_txt = VOICE_BANK / f'{profile_name}.txt'
    ref_text = ref_txt.read_text() if ref_txt.exists() else ''  # empty = auto-transcribe with Whisper
    out = OUT / f'{profile_name}_{time.time_ns()}.wav'
    get_tts().infer(ref_file=str(ref), ref_text=ref_text, gen_text=text, file_wave=str(out), speed=float(speed))
    return str(out)

with gr.Blocks(title='AI Music OS — Voice Clone') as demo:
    gr.Markdown('## 🎙️ Voice Clone (F5-TTS, free & local)\nUse only voices you have permission to clone.')
    with gr.Tab('1. Save a voice'):
        sample = gr.Audio(type='filepath', label='Reference voice sample (5-15 s)')
        ref_text = gr.Textbox(label='What is said in the sample (optional — leave empty to auto-detect)')
        profile = gr.Textbox(label='Profile name')
        save = gr.Button('Save profile', variant='primary')
        status = gr.Textbox(label='Status')
    with gr.Tab('2. Speak with it'):
        pick = gr.Dropdown(profiles(), label='Voice profile')
        text = gr.Textbox(lines=6, label='Text to speak')
        speed = gr.Slider(0.7, 1.3, value=1.0, step=0.05, label='Speed')
        gen = gr.Button('Generate', variant='primary')
        out = gr.Audio(label='Cloned speech', type='filepath')
    save.click(save_voice, [sample, profile, ref_text], [status, pick])
    gen.click(generate, [text, pick, speed], out)

demo.queue().launch(share=True, server_name='0.0.0.0', server_port=7867, allowed_paths=[str(OUT)])
