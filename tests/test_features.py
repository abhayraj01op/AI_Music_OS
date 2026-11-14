"""Feature behavior tests: real audio exports; model calls mocked unless noted."""
import ast
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import audio_tools

class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        self.bank = self.folder/'bank'; self.bank.mkdir()
        os.environ['AI_MUSIC_OUT'] = str(self.folder)
        os.environ['AI_MUSIC_VOICE_BANK'] = str(self.bank)
        self.k = importlib.reload(importlib.import_module('kokoro_app'))
        self.f = importlib.reload(importlib.import_module('f5_app'))
        sf.write(self.bank/'Alice.wav', np.ones(96000)*0.1, 24000)
        (self.bank/'Alice.txt').write_text('Reference words')
    def tearDown(self): self.tmp.cleanup()
    def fake_pipeline(self, text, **kwargs):
        return iter([('text','phonemes',np.ones(2400)*0.1)])
    def fake_tts(self):
        class TTS:
            seed = 42
            def infer(inner, **kwargs):
                inner.kwargs=kwargs
                sf.write(kwargs['file_wave'],np.ones(2400)*0.1,24000)
        return TTS()
    def test_catalog_languages_and_preserved_voices(self):
        self.assertEqual(len(self.k.VOICE_CHOICES),41)
        self.assertEqual(len(self.k.LANGUAGES),7)
        self.assertEqual(self.k.voices_for_language('Hindi'),['hf_alpha','hf_beta','hm_omega','hm_psi'])
        for voice in ['af_heart','af_bella','af_nicole','af_sarah','af_sky','am_adam','am_michael','bf_emma','bf_isabella','bm_george','bm_lewis']:
            self.assertIn(voice,self.k.VOICE_CHOICES)
    def test_long_text_is_not_truncated(self):
        text='x'*501
        chunks=audio_tools.split_text(text)
        self.assertEqual(''.join(chunks),text)
        self.assertLessEqual(max(map(len,chunks)),180)
    def test_pause_speed_and_subtitle_timing(self):
        calls=[]
        def pipeline(text,**kw): calls.append(kw);return self.fake_pipeline(text,**kw)
        with patch.object(self.k,'get_pipeline',return_value=pipeline):
            result=self.k.generate_audio('First sentence. Second sentence.','af_heart',1.25,0.5)
        self.assertEqual(calls[0]['speed'],1.25)
        self.assertEqual(sf.info(result).frames,16800)
        self.assertIn('00:00:00,600 --> 00:00:00,700',Path(result).with_suffix('.srt').read_text())
    def test_blend_weight_applied_to_real_tensor_argument(self):
        class P:
            def load_voice(inner,name):return np.array([1.0]) if name=='af_heart' else np.array([3.0])
            def __call__(inner,text,**kw):inner.voice=kw['voice'];return self.fake_pipeline(text)
        p=P()
        with patch.object(self.k,'get_pipeline',return_value=p):
            self.k.generate_audio('Hi','af_heart',blend_voice='af_bella',blend=0.25)
        self.assertEqual(p.voice.tolist(),[1.5])
    def test_cross_language_blend_rejected(self):
        with self.assertRaises(ValueError):self.k.generate_audio('Hi','af_heart',blend_voice='hf_alpha')
    def test_real_audio_exports_pitch_normalization(self):
        wav=self.folder/'test.wav'; sf.write(wav,np.sin(np.arange(24000)*0.1)*0.1,24000)
        for format in ('MP3','FLAC','WAV'):
            out=audio_tools.export_audio(wav,format,pitch=2,normalize=True)
            self.assertGreater(Path(out).stat().st_size,100)
            self.assertLess(abs(sf.info(out).duration-1),0.1)
    def test_batch_zip_contains_distinct_audio_and_metadata(self):
        with patch.object(self.k,'get_pipeline',return_value=self.fake_pipeline):
            result=self.k.batch_generate('one\ntwo','af_heart')
        with zipfile.ZipFile(result) as z:
            self.assertEqual(len([n for n in z.namelist() if n.endswith('.wav')]),2)
            self.assertEqual(len([n for n in z.namelist() if n.endswith('.srt')]),2)
            self.assertEqual(len([n for n in z.namelist() if n.endswith('.json')]),2)
    def test_f5_quality_seed_and_export_forwarded(self):
        tts=self.fake_tts()
        with patch.object(self.f,'get_tts',return_value=tts):
            result=self.f.generate('Hello','Alice',1.2,64,42,3.1,0.25,True,'FLAC')
        self.assertEqual(tts.kwargs['nfe_step'],64)
        self.assertEqual(tts.kwargs['seed'],42)
        self.assertEqual(tts.kwargs['cfg_strength'],3.1)
        self.assertEqual(tts.kwargs['cross_fade_duration'],0.25)
        self.assertTrue(tts.kwargs['remove_silence'])
        self.assertEqual(json.loads(Path(result).with_suffix('.json').read_text())['seed'],42)
    def test_profile_trim_edit_and_overwrite_guard(self):
        sample=self.folder/'sample.wav';sf.write(sample,np.sin(np.arange(24000*8)*0.1)*0.1,24000)
        self.f.save_voice(sample,'Bob','Words',start=2,duration=4)
        self.assertEqual(sf.info(self.bank/'Bob.wav').frames,96000)
        self.f.update_transcript('Bob','Corrected');self.assertEqual(self.f.load_profile('Bob')[1],'Corrected')
        with self.assertRaises(Exception):self.f.save_voice(sample,'Bob','Oops')
        self.assertEqual((self.bank/'Bob.txt').read_text(),'Corrected')
    def test_short_or_silent_reference_rejected(self):
        sample=self.folder/'silent.wav';sf.write(sample,np.zeros(96000),24000)
        with self.assertRaises(Exception):self.f.save_voice(sample,'Silent','')
        self.assertFalse((self.bank/'Silent.wav').exists())
    def test_dialogue_joins_profiles_with_pause(self):
        sf.write(self.bank/'Bob.wav',np.ones(96000)*0.1,24000)
        tts=self.fake_tts()
        with patch.object(self.f,'get_tts',return_value=tts):
            path,_=self.f.dialogue_generate('Alice: Hello\nBob: Welcome')
        self.assertEqual(sf.info(path).frames,10800)
    def test_embedded_apps_helpers_and_env_match(self):
        for prefix,app in [('02','kokoro'),('03','f5')]:
            path=next((ROOT/'colab').glob(prefix+'*.ipynb'))
            tree=ast.parse(''.join(json.loads(path.read_text())['cells'][4]['source']))
            strings=[n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
            self.assertIn((ROOT/(app+'_app.py')).read_text(),strings)
            self.assertIn((ROOT/'audio_tools.py').read_text(),strings)
            ns={'WORK':self.folder,'ENV':{},'OUT':self.folder,'VOICE_BANK':self.bank,'PY':Path('/python')}
            exec(compile(tree,'embed','exec'),ns)
            self.assertEqual(ns['ENV']['AI_MUSIC_OUT'],str(self.folder))
    def test_both_interfaces_construct(self):
        self.assertGreater(len(self.k.build_demo().config['dependencies']),3)
        self.assertGreater(len(self.f.build_demo().config['dependencies']),7)

if __name__=='__main__':unittest.main()
