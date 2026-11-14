"""CPU-only regression checks; no model downloads or Google Drive needed."""
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'colab/02_Kokoro_Colab.ipynb'

class KokoroInstallTests(unittest.TestCase):
    def run_installer(self, version):
        source = ''.join(json.loads(NOTEBOOK.read_text())['cells'][3]['source'])
        with patch.object(sys, 'version_info', version), patch.object(metadata, 'version', return_value='1.0'), patch.object(subprocess, 'run') as run:
            exec(compile(source, str(NOTEBOOK), 'exec'), {})
        return [call.args[0] for call in run.call_args_list]

    def test_python_313_resolves_english_dependencies(self):
        commands = self.run_installer((3, 13, 0))
        install = commands[0]
        self.assertIn('misaki[en] @ git+https://github.com/hexgrad/misaki.git@fba1236595f2d2bf21d414ba6e57d25256afada3', install)
        self.assertNotIn('--no-deps', install)
        self.assertIn('from misaki.token import MToken', commands[1][2])

    def test_python_312_uses_complete_install_too(self):
        self.assertEqual(self.run_installer((3, 12, 0))[0], self.run_installer((3, 13, 0))[0])

    def test_core_packages_are_constrained(self):
        commands = self.run_installer((3, 13, 0))
        install = commands[0]
        constraints = Path(install[install.index('--constraint') + 1]).read_text()
        self.assertEqual(constraints, 'numpy==1.0\nscipy==1.0\ntorch==1.0\ntorchaudio==1.0\ntorchvision==1.0\n')

    def test_unsupported_python_fails_before_install(self):
        with self.assertRaises(RuntimeError):
            self.run_installer((3, 14, 0))

    def test_all_music_code_cells_compile(self):
        for path in (ROOT / 'colab').glob('*.ipynb'):
            for index, cell in enumerate(json.loads(path.read_text())['cells']):
                if cell['cell_type'] == 'code':
                    compile(''.join(cell['source']), f'{path.name}:{index}', 'exec')

if __name__ == '__main__':
    unittest.main()
