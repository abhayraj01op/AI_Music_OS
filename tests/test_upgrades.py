import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model_config import ACE_MODELS,QWEN_MODELS,ace_command,qwen_settings
class UpgradeTests(unittest.TestCase):
    def test_full_qwen_models(self):
        self.assertIn('Qwen/Qwen3-TTS-12Hz-1.7B-Base',QWEN_MODELS)
        self.assertIn('Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice',QWEN_MODELS)
        self.assertIn('Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign',QWEN_MODELS)
        self.assertEqual(len(QWEN_MODELS),5)
    def test_t4_qwen_no_flash_or_bf16(self):
        settings=qwen_settings(QWEN_MODELS[0]);self.assertEqual(settings['dtype'],'float16');self.assertEqual(settings['attn_implementation'],'sdpa')
    def test_full_ace_not_service_restricted(self):
        self.assertEqual(len(ACE_MODELS),6)
        cmd=ace_command('ace','acestep-v15-xl-base','acestep-5Hz-lm-4B','/outputs')
        self.assertEqual(cmd[cmd.index('--service_mode')+1],'false')
        self.assertEqual(cmd[cmd.index('--config_path')+1],'acestep-v15-xl-base')
        self.assertEqual(cmd[cmd.index('--lm_model_path')+1],'acestep-5Hz-lm-4B')
        self.assertEqual(cmd[cmd.index('--offload_to_cpu')+1],'true')
        self.assertEqual(cmd[cmd.index('--backend')+1],'pt')
    def test_no_planner(self):
        cmd=ace_command('ace','acestep-v15-turbo','None','/outputs')
        self.assertEqual(cmd[cmd.index('--init_llm')+1],'false');self.assertNotIn('--lm_model_path',cmd)
    def test_reject_invalid_models(self):
        with self.assertRaises(ValueError):qwen_settings('fake')
        with self.assertRaises(ValueError):ace_command('ace','fake','None','/outputs')
    def test_embedded_qwen_sources_and_env(self):
        j=json.loads((ROOT/'colab/03_Qwen_Voice_Clone_Colab.ipynb').read_text())
        with tempfile.TemporaryDirectory() as folder:
            ns=dict(ENV={},WORK=Path(folder),OUT=Path(folder),VOICE_BANK=Path(folder),PY=Path('python'))
            exec(''.join(j['cells'][4]['source']),ns)
            for name in ['qwen_app.py','model_config.py','audio_tools.py','f5_app.py']:
                self.assertEqual(Path(folder,name).read_text(),(ROOT/name).read_text())
            self.assertEqual(ns['ENV']['AI_MUSIC_QWEN_MODEL'],'Qwen/Qwen3-TTS-12Hz-1.7B-Base')
            self.assertEqual(ns['ENV']['GRADIO_TEMP_DIR'],str(Path(folder)/'qwen_generated'))
    def test_ace_frozen_lock_and_managed_python(self):
        j=json.loads((ROOT/'colab/01_ACE_Step_Colab.ipynb').read_text());tree=ast.parse(''.join(j['cells'][2]['source']));ns={}
        for node in tree.body[:3]:exec(compile(ast.Module(body=[node],type_ignores=[]),'lock','exec'),ns)
        self.assertEqual(ns['LOCK_CONTENT'],(ROOT/'requirements/ace15.uv.lock').read_text())
        self.assertIn("'--frozen'",''.join(j['cells'][2]['source']))
        self.assertIn('3.12.16',''.join(j['cells'][1]['source']))
    def test_all_cells_compile(self):
        for path in (ROOT/'colab').glob('*.ipynb'):
            for c in json.loads(path.read_text())['cells']:
                if c['cell_type']=='code':compile(''.join(c['source']),str(path),'exec')
if __name__=='__main__':unittest.main()
