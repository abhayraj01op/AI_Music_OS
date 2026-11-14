from pathlib import Path
import os
import time
import numpy as np
import soundfile as sf
import gradio as gr
from kokoro import KPipeline
from audio_tools import split_text, export_audio, save_metadata, package_files, read_text_file, recent_outputs, timestamp

OUT = Path(os.environ['AI_MUSIC_OUT'])
OUT.mkdir(parents=True, exist_ok=True)
LANGUAGES = {'American English':'a', 'British English':'b', 'Hindi':'h', 'Spanish':'e', 'French':'f', 'Italian':'i', 'Brazilian Portuguese':'p'}
VOICE_CHOICES = 'af_alloy af_aoede af_bella af_heart af_jessica af_kore af_nicole af_nova af_river af_sarah af_sky am_adam am_echo am_eric am_fenrir am_liam am_michael am_onyx am_puck am_santa bf_alice bf_emma bf_isabella bf_lily bm_daniel bm_fable bm_george bm_lewis ef_dora em_alex em_santa ff_siwis hf_alpha hf_beta hm_omega hm_psi if_sara im_nicola pf_dora pm_alex pm_santa'.split()
SAMPLE_RATE = 24000
pipelines = {}


def get_pipeline(code):
    if code not in pipelines:
        if pipelines:
            pipelines[code] = KPipeline(lang_code=code, model=next(iter(pipelines.values())).model)
        else:
            pipelines[code] = KPipeline(lang_code=code)
    return pipelines[code]


def voices_for_language(language):
    code = LANGUAGES[language]
    return [v for v in VOICE_CHOICES if v.startswith(code)]


def change_language(language):
    choices = voices_for_language(language)
    return gr.update(choices=choices, value=choices[0]), gr.update(choices=['None'] + choices, value='None')


def generate_audio(text, voice='af_heart', speed=1.0, pause=0.15, blend_voice='None', blend=0.5, format='WAV', pitch=0, normalize=False):
    text = (text or '').strip()
    if not text:
        raise ValueError('Text is empty.')
    if voice not in VOICE_CHOICES:
        raise ValueError(f'Unsupported voice: {voice}')
    if not 0.5 <= float(speed) <= 2 or not 0 <= float(pause) <= 2:
        raise ValueError('Speed must be 0.5–2; pause must be 0–2 seconds.')
    if blend_voice not in (None, 'None') and (blend_voice not in VOICE_CHOICES or blend_voice[0] != voice[0]):
        raise ValueError('Blend voices must use the same language/accent.')
    if not 0 <= float(blend) <= 1:
        raise ValueError('Blend weight must be 0–1.')
    selected = get_pipeline(voice[0])
    effective_voice = voice
    if blend_voice not in (None, 'None'):
        effective_voice = selected.load_voice(voice) * (1 - float(blend)) + selected.load_voice(blend_voice) * float(blend)
    chunks, subtitles, cursor = [], [], 0.0
    for part in split_text(text):
        part_audio = []
        for item in selected(part, voice=effective_voice, speed=float(speed)):
            audio = item[2] if isinstance(item, tuple) and len(item) >= 3 else item.get('audio') if isinstance(item, dict) else getattr(item, 'audio', None)
            if audio is None:
                raise RuntimeError(f'Unsupported Kokoro output type: {type(item)}')
            if hasattr(audio, 'detach'):
                audio = audio.detach().cpu().numpy()
            array = np.asarray(audio, dtype=np.float32).reshape(-1)
            if array.size == 0 or not np.isfinite(array).all():
                raise RuntimeError('Invalid Kokoro audio.')
            part_audio.append(array)
        if not part_audio:
            raise RuntimeError('Kokoro returned no audio.')
        joined = np.concatenate(part_audio)
        if chunks:
            chunks.append(np.zeros(round(float(pause) * SAMPLE_RATE), dtype=np.float32))
            cursor += float(pause)
        start = cursor
        chunks.append(joined)
        cursor += len(joined) / SAMPLE_RATE
        subtitles.append(f'{len(subtitles)+1}\n{timestamp(start)} --> {timestamp(cursor)}\n{part}\n')
    if not chunks:
        raise RuntimeError('Kokoro returned no audio.')
    wav = OUT / f'kokoro_{time.time_ns()}.wav'
    sf.write(wav, np.concatenate(chunks), SAMPLE_RATE)
    result = export_audio(wav, format, pitch, normalize)
    Path(result).with_suffix('.srt').write_text('\n'.join(subtitles), encoding='utf-8')
    save_metadata(result, dict(text=text, voice=voice, blend_voice=blend_voice, blend=blend, speed=speed, pause=pause, pitch=pitch, normalize=normalize, duration=cursor, subtitles='chunk-level approximate timing'))
    return result


def ui_generate(*args):
    try:
        result = generate_audio(*args)
        return result, result, str(Path(result).with_suffix('.srt')), f'{sf.info(result).duration:.2f} seconds'
    except Exception as e:
        raise gr.Error(str(e)) from e


def batch_generate(text, *settings):
    lines = [line.strip() for line in (text or '').splitlines() if line.strip()]
    if not lines or len(lines) > 50:
        raise gr.Error('Enter 1–50 nonempty lines, one clip per line.')
    try:
        files = [generate_audio(line, *settings) for line in lines]
        return package_files(files, OUT, 'kokoro_batch')
    except Exception as e:
        raise gr.Error(str(e)) from e


def build_demo():
    with gr.Blocks(title='AI Music OS — Kokoro v7') as demo:
        gr.Markdown('## Kokoro TTS — 41 voices · 7 languages')
        with gr.Row():
            language = gr.Dropdown(list(LANGUAGES), value='American English', label='Language / accent', min_width=240)
            voice = gr.Dropdown(voices_for_language('American English'), value='af_heart', label='Voice', min_width=240)
        with gr.Row():
            speed = gr.Slider(0.5, 2, value=1, step=0.05, label='Speech speed', min_width=240)
            pause = gr.Slider(0, 2, value=0.15, step=0.05, label='Pause between chunks (seconds)', min_width=240)
        with gr.Accordion('Voice blend & export', open=False):
            second = gr.Dropdown(['None'] + voices_for_language('American English'), value='None', label='Second voice')
            blend = gr.Slider(0, 1, value=0.5, step=0.05, label='Second voice weight')
            format = gr.Radio(['WAV', 'MP3', 'FLAC'], value='WAV', label='Export format')
            pitch = gr.Slider(-6, 6, value=0, step=0.5, label='Pitch shift (semitones)')
            normalize = gr.Checkbox(label='Normalize loudness', value=False)
        settings = [voice, speed, pause, second, blend, format, pitch, normalize]
        with gr.Tab('Speak'):
            text = gr.Textbox(lines=8, label='Text', value='Hello. Welcome to Music OS.')
            upload = gr.File(file_types=['.txt'], label='Import text', type='filepath')
            gen = gr.Button('Generate speech', variant='primary')
            audio = gr.Audio(type='filepath', label='Speech')
            download = gr.File(label='Download audio')
            srt = gr.File(label='Subtitles (approximate chunk timing)')
            info = gr.Textbox(label='Duration')
        with gr.Tab('Batch clips'):
            batch = gr.Textbox(lines=10, label='One clip per line')
            batch_button = gr.Button('Generate batch', variant='primary')
            archive = gr.File(label='Clips, settings & subtitles ZIP')
        with gr.Tab('Saved outputs'):
            refresh = gr.Button('Refresh')
            history = gr.Files(label='Recent downloads')
        language.change(change_language, language, [voice, second])
        upload.change(read_text_file, upload, text)
        gen.click(ui_generate, [text] + settings, [audio, download, srt, info])
        batch_button.click(batch_generate, [batch] + settings, archive)
        refresh.click(lambda: recent_outputs(OUT), outputs=history)
    return demo


if __name__ == '__main__':
    print('Kokoro smoke test:', generate_audio('Hello. This is Music OS.', 'af_heart'))
    build_demo().queue(default_concurrency_limit=1).launch(share=True, server_name='0.0.0.0', server_port=7866, allowed_paths=[str(OUT)])
