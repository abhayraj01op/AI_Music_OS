"""Run with Python in a test environment containing Gradio, NumPy, SoundFile and Torch.
CPU regression checks; F5 inference is mocked, not represented as GPU verification.
"""
import ast
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import soundfile as sf
import torch
import gradio as gr
ROOT = Path(__file__).resolve().parents[1]

def definitions(name):
    source = (ROOT / f'{name}_app.py').read_text()
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    namespace = {'np':np, 'sf':sf, 'gr':gr, 'torch':torch, 'subprocess':subprocess}
    import time
    namespace['time']=time
    exec(compile(ast.Module(body=nodes,type_ignores=[]),name,'exec'),namespace)
    return namespace

class MusicTests(unittest.TestCase):
    def test_notebook_code_compiles(self):
        for path in (ROOT/'colab').glob('*.ipynb'):
            for i,c in enumerate(json.loads(path.read_text())['cells']):
                if c['cell_type']=='code': compile(''.join(c['source']),f'{path}:{i}','exec')

    def test_isolated_install_does_not_constrain_broken_kernel(self):
        for path in (ROOT/'colab').glob('*.ipynb'):
            cells=json.loads(path.read_text())['cells']
            namespace={}
            install=ast.parse(''.join(cells[2]['source']))
            for node in install.body[:2]: exec(compile(ast.Module(body=[node],type_ignores=[]),'lock','exec'),namespace)
            self.assertIn('torch==2.6.0+cu124',namespace['LOCK_CONTENT'])
            self.assertIn('torchaudio==2.6.0+cu124',namespace['LOCK_CONTENT'])
            commands=[]
            with tempfile.TemporaryDirectory() as folder:
                namespace.update(WORK=Path(folder),UV=Path('/isolated/bin/uv'),PY=Path('/isolated/bin/python'),run_logged=lambda cmd: commands.append(cmd))
                exec(compile(ast.Module(body=install.body[2:],type_ignores=[]),'install','exec'),namespace)
                self.assertEqual(commands[0][1:3],['pip','sync'])
                self.assertEqual(commands[0][commands[0].index('--python')+1],'/isolated/bin/python')
                self.assertEqual(commands[1][1:3],['pip','check'])
                self.assertNotIn('--no-deps',commands[0])

    def test_complete_kokoro_phonemizer(self):
        lock=(ROOT/'requirements/kokoro.lock').read_text()
        self.assertIn('addict==2.4.0',lock)
        self.assertIn('pip==25.1.1',lock)
        self.assertIn('en-core-web-sm @ https://github.com/explosion/spacy-models/',lock)

    def test_ace_t4_flags_match_cli(self):
        cells=json.loads((ROOT/'colab/01_ACE_Step_Colab.ipynb').read_text())['cells']
        ns={'VENV':Path('/env'),'MODELS':Path('/models'),'run_logged':lambda cmd: None}
        exec(''.join(cells[4]['source']),ns)
        cmd=ns['COMMAND']
        self.assertEqual(cmd[cmd.index('--bf16')+1],'false')
        self.assertEqual(cmd[cmd.index('--cpu_offload')+1],'true')
        self.assertEqual(cmd[cmd.index('--overlapped_decode')+1],'true')

    def test_kokoro_cuda_tensor_conversion_british_and_unique_wav(self):
        ns=definitions('kokoro')
        calls=[]
        class DeviceAudio:
            def detach(self): calls.append('detach'); return self
            def cpu(self): calls.append('cpu'); return self
            def numpy(self): return np.ones(4800,dtype=np.float32)*0.1
        with tempfile.TemporaryDirectory() as folder:
            ns.update(OUT=Path(folder),SAMPLE_RATE=24000,VOICE_CHOICES=['af_heart','bf_emma'],pipelines={'a':lambda *a,**kw: iter([('g','p',DeviceAudio())]),'b':lambda *a,**kw: iter([('g','p',DeviceAudio())])})
            a=ns['generate_audio']('hello','af_heart'); b=ns['generate_audio']('hello','bf_emma')
            self.assertNotEqual(a,b)
            self.assertEqual(calls,['detach','cpu','detach','cpu'])
            wav,sr=sf.read(b); self.assertEqual(sr,24000); self.assertEqual(len(wav),4800)

    def test_empty_kokoro_text_rejected(self):
        with self.assertRaises(ValueError): definitions('kokoro')['generate_audio'](' ')

    def test_f5_profile_real_ffmpeg_and_mock_infer_readback(self):
        ns=definitions('f5')
        with tempfile.TemporaryDirectory() as folder:
            bank=Path(folder)/'bank'; bank.mkdir(); out=Path(folder)/'out'; out.mkdir()
            sample=Path(folder)/'source.wav'; sf.write(sample,np.sin(np.arange(44100)*0.1)*0.1,44100)
            ns.update(VOICE_BANK=bank,OUT=out)
            ns['save_voice'](sample,'Alice','Reference words')
            wav,sr=sf.read(bank/'Alice.wav'); self.assertEqual(sr,24000)
            self.assertEqual((bank/'Alice.txt').read_text(),'Reference words')
            class FakeTTS:
                def infer(self,**kwargs):
                    self.kwargs=kwargs; sf.write(kwargs['file_wave'],np.ones(2400)*0.1,24000)
            fake=FakeTTS(); ns['get_tts']=lambda:fake
            result=ns['generate']('Hello','Alice',1.0)
            self.assertEqual(fake.kwargs['ref_text'],'Reference words')
            self.assertTrue(Path(result).exists())
            self.assertEqual(sf.info(result).samplerate,24000)

    def test_f5_unsaved_profile_is_rejected(self):
        ns=definitions('f5')
        with tempfile.TemporaryDirectory() as folder:
            ns['VOICE_BANK']=Path(folder)
            with self.assertRaises(gr.Error): ns['generate']('hello','../outside',1.0)

if __name__=='__main__': unittest.main()
