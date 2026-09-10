import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from pi_trainer.compiler import run_compile

class CompilerTests(unittest.TestCase):
    def test_genai_missing_recipe_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'vendor-qualified'):
                run_compile({'task':'llm','target':'hailo10h','model':tmp},Path(tmp),lambda _:None,lambda:False)
    def test_no_calibration_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            model=Path(tmp)/'model.onnx';model.write_bytes(b'x')
            with self.assertRaisesRegex(ValueError,'calibration_path'):
                run_compile({'task':'vision','target':'hailo8','model':str(model)},Path(tmp),lambda _:None,lambda:False)
    def test_wrong_target_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model=root/'m.onnx';model.write_bytes(b'x');cal=root/'c.npy';cal.write_bytes(b'x')
            (root/'compile-result.json').write_text(json.dumps({'target':'hailo8','artifact_paths':['m.hef']}))
            with patch('pi_trainer.processes.run_process'):
                with self.assertRaisesRegex(ValueError,'wrong target'):
                    run_compile({'task':'vision','target':'hailo8l','model':str(model),'calibration_path':str(cal)},root,lambda _:None,lambda:False)
    def test_outputs_cannot_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model=root/'m.onnx';model.write_bytes(b'x');cal=root/'c.npy';cal.write_bytes(b'x')
            (root/'compile-result.json').write_text(json.dumps({'target':'hailo8','artifact_paths':['../other.hef']}))
            with patch('pi_trainer.processes.run_process'):
                with self.assertRaisesRegex(ValueError,'escaped'):
                    run_compile({'task':'vision','target':'hailo8','model':str(model),'calibration_path':str(cal)},root,lambda _:None,lambda:False)
