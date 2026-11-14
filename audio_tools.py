"""Shared local audio exports; embedded beside each standalone notebook app."""
from pathlib import Path
import json
import math
import subprocess
import time
import zipfile
import numpy as np
import soundfile as sf


def read_text_file(path):
    if not path:
        raise ValueError('Choose a UTF-8 .txt file.')
    p = Path(path)
    if p.suffix.lower() != '.txt' or p.stat().st_size > 1_000_000:
        raise ValueError('Use a .txt file smaller than 1 MB.')
    return p.read_text(encoding='utf-8-sig')


def split_text(text, limit=180):
    """Bound chunks even for punctuation-free text (Espeak otherwise truncates)."""
    import re
    chunks = []
    for line in (text or '').strip().splitlines():
        for part in re.split(r'(?<=[.!?。！？।])\s*', line.strip()):
            while len(part) > limit:
                cut = part.rfind(' ', 0, limit + 1)
                cut = cut if cut > 0 else limit
                chunks.append(part[:cut].strip())
                part = part[cut:].strip()
            if part:
                chunks.append(part)
    return chunks


def export_audio(wav, format='WAV', pitch=0, normalize=False):
    if format not in ('WAV', 'MP3', 'FLAC'):
        raise ValueError('Choose WAV, MP3 or FLAC.')
    if not math.isfinite(float(pitch)) or not -6 <= float(pitch) <= 6:
        raise ValueError('Pitch must be between -6 and +6 semitones.')
    wav = Path(wav)
    if format == 'WAV' and not pitch and not normalize:
        return str(wav)
    target = wav.with_name(wav.stem + '_export.' + format.lower())
    filters = []
    if pitch:
        ratio = 2 ** (float(pitch) / 12)
        filters += [f'asetrate={sf.info(wav).samplerate}*{ratio}', 'aresample=24000', f'atempo={1 / ratio}']
    if normalize:
        filters.append('loudnorm=I=-16:TP=-1.5:LRA=11')
    cmd = ['ffmpeg', '-y', '-v', 'error', '-i', str(wav)]
    if filters:
        cmd += ['-af', ','.join(filters)]
    if format == 'MP3':
        cmd += ['-codec:a', 'libmp3lame', '-b:a', '192k']
    subprocess.run(cmd + [str(target)], check=True)
    return str(target)


def save_metadata(wav, data):
    Path(wav).with_suffix('.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def package_files(files, out, prefix):
    path = Path(out) / f'{prefix}_{time.time_ns()}.zip'
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for f in files:
            p = Path(f)
            archive.write(p, p.name)
            for suffix in ('.json', '.srt'):
                sidecar = p.with_suffix(suffix)
                if sidecar.exists():
                    archive.write(sidecar, sidecar.name)
    return str(path)


def recent_outputs(out):
    files = sorted((p for p in Path(out).iterdir() if p.suffix.lower() in ('.wav', '.mp3', '.flac', '.zip')), key=lambda p: p.stat().st_mtime, reverse=True)[:30]
    return [str(p) for p in files]


def timestamp(seconds):
    ms = round(seconds * 1000)
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    sec, ms = divmod(ms, 1000)
    return f'{hours:02}:{minutes:02}:{sec:02},{ms:03}'
