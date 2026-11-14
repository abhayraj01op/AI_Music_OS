from pathlib import Path
import os
import time
import subprocess
import numpy as np
import soundfile as sf
import gradio as gr
from audio_tools import export_audio, save_metadata, package_files, read_text_file, recent_outputs
OUT = Path(os.environ['AI_MUSIC_OUT'])
VOICE_BANK = Path(os.environ['AI_MUSIC_VOICE_BANK'])
OUT.mkdir(parents=True, exist_ok=True)
VOICE_BANK.mkdir(parents=True, exist_ok=True)
_tts = None


def get_tts():
    global _tts
    if _tts is None:
        from f5_tts.api import F5TTS
        print('Loading F5-TTS (first run downloads model to Drive)...')
        _tts = F5TTS()
    return _tts


def profiles():
    return sorted(p.stem for p in VOICE_BANK.glob('*.wav'))


def profile_paths(name):
    if not name or name not in profiles():
        raise gr.Error('Choose a saved voice profile.')
    return VOICE_BANK / f'{name}.wav', VOICE_BANK / f'{name}.txt'


def load_profile(name):
    ref, txt = profile_paths(name)
    return str(ref), txt.read_text(encoding='utf-8') if txt.exists() else ''


def save_voice(sample, profile_name, ref_text, start=0, duration=15, normalize=False, overwrite=False):
    if sample is None:
        raise gr.Error('Upload or record a voice sample first.')
    safe = ''.join(c for c in (profile_name or '') if c.isalnum() or c in (' ', '_', '-')).strip()
    if not safe:
        raise gr.Error('Enter a profile name.')
    if not 0 <= float(start) or not 3 <= float(duration) <= 15:
        raise gr.Error('Choose a nonnegative start and 3–15 seconds of reference speech.')
    dst = VOICE_BANK / f'{safe}.wav'
    if dst.exists() and not overwrite:
        raise gr.Error('Profile exists; enable overwrite or choose another name.')
    temp = VOICE_BANK / f'_reference_{time.time_ns()}.wav'
    cmd = ['ffmpeg', '-y', '-v', 'error', '-ss', str(start), '-i', str(sample), '-t', str(duration), '-ac', '1', '-ar', '24000']
    if normalize:
        cmd += ['-af', 'loudnorm=I=-20:TP=-2:LRA=11']
    try:
        subprocess.run(cmd + [str(temp)], check=True)
        audio, sr = sf.read(temp)
        if len(audio) < 3 * sr or not np.isfinite(audio).all() or np.max(np.abs(audio)) < 0.0001:
            raise gr.Error('Selected reference must contain at least 3 seconds of audible speech.')
        temp.replace(dst)
    finally:
        temp.unlink(missing_ok=True)
    (VOICE_BANK / f'{safe}.txt').write_text((ref_text or '').strip(), encoding='utf-8')
    return f'Saved: {safe}', gr.update(choices=profiles(), value=safe)


def update_transcript(name, text):
    _, path = profile_paths(name)
    path.write_text((text or '').strip(), encoding='utf-8')
    return 'Transcript saved.'


def generate(text, profile_name, speed=1.0, steps=32, seed=-1, cfg=2.0, crossfade=0.15, remove_silence=False, format='WAV', pitch=0, normalize=False):
    if not (text or '').strip():
        raise gr.Error('Enter text.')
    ref, ref_txt = profile_paths(profile_name)
    if not 0.5 <= float(speed) <= 2 or int(steps) not in (16, 32, 64):
        raise gr.Error('Speed must be 0.5–2; choose 16, 32 or 64 sampling steps.')
    if float(seed) != int(seed) or int(seed) < -1 or int(seed) > 2147483647:
        raise gr.Error('Seed must be -1 (random) or an integer from 0 to 2147483647.')
    if not 0 <= float(cfg) <= 5 or not 0 <= float(crossfade) <= 0.5:
        raise gr.Error('Guidance must be 0–5; crossfade must be 0–0.5 seconds.')
    ref_text = ref_txt.read_text(encoding='utf-8') if ref_txt.exists() else ''
    out = OUT / f'{profile_name}_{time.time_ns()}.wav'
    tts = get_tts()
    tts.infer(ref_file=str(ref), ref_text=ref_text, gen_text=text.strip(), file_wave=str(out), speed=float(speed), nfe_step=int(steps), seed=None if int(seed) == -1 else int(seed), cfg_strength=float(cfg), cross_fade_duration=float(crossfade), remove_silence=bool(remove_silence))
    if not out.exists() or sf.info(out).frames == 0:
        raise RuntimeError('F5-TTS did not save audio.')
    result = export_audio(out, format, pitch, normalize)
    save_metadata(result, dict(text=text, profile=profile_name, speed=speed, steps=steps, seed=getattr(tts, 'seed', seed), cfg=cfg, crossfade=crossfade, remove_silence=remove_silence, pitch=pitch, normalize=normalize))
    return result


def ui_generate(*args):
    try:
        result = generate(*args)
        return result, result, Path(result).with_suffix('.json').read_text(encoding='utf-8')
    except Exception as e:
        raise gr.Error(str(e)) from e


def batch_generate(text, profile_name, *settings):
    lines = [line.strip() for line in (text or '').splitlines() if line.strip()]
    if not lines or len(lines) > 50:
        raise gr.Error('Enter 1–50 nonempty lines, one clip per line.')
    try:
        return package_files([generate(line, profile_name, *settings) for line in lines], OUT, 'f5_batch')
    except Exception as e:
        raise gr.Error(str(e)) from e


def dialogue_generate(script, speed=1):
    lines = [line.strip() for line in (script or '').splitlines() if line.strip()]
    if not lines or len(lines) > 50:
        raise gr.Error('Enter 1–50 lines as Profile name: words to speak.')
    turns = []
    for line in lines:
        name, sep, words = line.partition(':')
        if not sep or not words.strip():
            raise gr.Error('Every dialogue line must be Profile name: words to speak.')
        profile_paths(name.strip())
        turns.append((name.strip(), words.strip()))
    try:
        parts, sr = [], None
        for name, words in turns:
            path = generate(words, name, speed)
            audio, rate = sf.read(path)
            if sr is not None and sr != rate:
                raise RuntimeError('Dialogue sample rates do not match.')
            sr = rate
            if parts:
                parts.append(np.zeros(round(rate * 0.25)))
            parts.append(audio)
        out = OUT / f'dialogue_{time.time_ns()}.wav'
        sf.write(out, np.concatenate(parts), sr)
        save_metadata(out, dict(script=script, speed=speed, pause=0.25))
        return str(out), str(out)
    except Exception as e:
        raise gr.Error(str(e)) from e


def build_demo():
    with gr.Blocks(title='AI Music OS — F5-TTS v7') as demo:
        gr.Markdown('## F5-TTS Voice Clone · English / Chinese\nClone only voices you have permission to use.')
        pick = gr.Dropdown(profiles(), label='Voice profile')
        refresh = gr.Button('Refresh profiles')
        with gr.Tab('Save a voice'):
            sample = gr.Audio(sources=['upload', 'microphone'], type='filepath', label='Reference speech')
            ref_text = gr.Textbox(label='Reference transcript (blank = automatic transcription)')
            name = gr.Textbox(label='Profile name')
            start = gr.Number(value=0, minimum=0, label='Reference start (seconds)')
            duration = gr.Slider(3, 15, value=15, step=0.5, label='Reference length (seconds)')
            ref_norm = gr.Checkbox(label='Normalize reference loudness')
            overwrite = gr.Checkbox(label='Overwrite existing profile', value=False)
            save = gr.Button('Save profile', variant='primary')
            status = gr.Textbox(label='Status')
        with gr.Tab('Preview & edit profile'):
            preview = gr.Audio(type='filepath', label='Saved reference')
            transcript = gr.Textbox(lines=4, label='Saved transcript')
            edit = gr.Button('Save transcript')
            edit_status = gr.Textbox(label='Status')
        with gr.Tab('Speak & batch'):
            text = gr.Textbox(lines=8, label='Text to speak')
            upload = gr.File(file_types=['.txt'], type='filepath', label='Import text')
            speed = gr.Slider(0.5, 2, value=1, step=0.05, label='Speed')
            with gr.Accordion('Quality & export', open=False):
                steps = gr.Radio([16, 32, 64], value=32, label='Sampling steps')
                seed = gr.Number(value=-1, precision=0, label='Seed (-1 = random)')
                cfg = gr.Slider(0, 5, value=2, step=0.1, label='Guidance')
                crossfade = gr.Slider(0, 0.5, value=0.15, step=0.01, label='Chunk crossfade (seconds)')
                silence = gr.Checkbox(label='Remove generated silence')
                format = gr.Radio(['WAV', 'MP3', 'FLAC'], value='WAV', label='Export')
                pitch = gr.Slider(-6, 6, value=0, step=0.5, label='Pitch shift (semitones)')
                normalize = gr.Checkbox(label='Normalize output loudness')
            settings = [speed, steps, seed, cfg, crossfade, silence, format, pitch, normalize]
            gen = gr.Button('Generate', variant='primary')
            audio = gr.Audio(type='filepath', label='Cloned speech')
            download = gr.File(label='Download audio')
            metadata = gr.Textbox(lines=4, label='Generation settings & seed')
            batch = gr.Textbox(lines=6, label='Batch: one clip per line')
            batch_button = gr.Button('Generate batch')
            archive = gr.File(label='Batch ZIP')
        with gr.Tab('Multi-voice dialogue'):
            script = gr.Textbox(lines=10, label='Profile name: words to speak (one turn per line)')
            dialogue_speed = gr.Slider(0.5, 2, value=1, step=0.05, label='Dialogue speed')
            dialogue_button = gr.Button('Generate dialogue', variant='primary')
            dialogue_audio = gr.Audio(type='filepath', label='Dialogue')
            dialogue_download = gr.File(label='Download dialogue')
        with gr.Tab('Saved outputs'):
            output_refresh = gr.Button('Refresh outputs')
            history = gr.Files(label='Recent downloads')
        refresh.click(lambda: gr.update(choices=profiles()), outputs=pick)
        save.click(save_voice, [sample, name, ref_text, start, duration, ref_norm, overwrite], [status, pick])
        pick.change(load_profile, pick, [preview, transcript])
        edit.click(update_transcript, [pick, transcript], edit_status)
        upload.change(read_text_file, upload, text)
        gen.click(ui_generate, [text, pick] + settings, [audio, download, metadata])
        batch_button.click(batch_generate, [batch, pick] + settings, archive)
        dialogue_button.click(dialogue_generate, [script, dialogue_speed], [dialogue_audio, dialogue_download])
        output_refresh.click(lambda: recent_outputs(OUT), outputs=history)
    return demo


if __name__ == '__main__':
    build_demo().queue(default_concurrency_limit=1).launch(share=True, server_name='0.0.0.0', server_port=7867, allowed_paths=[str(OUT), str(VOICE_BANK)])
