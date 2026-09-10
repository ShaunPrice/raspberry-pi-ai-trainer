import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from pi_trainer.training import run_training,validate,artifact

class TrainingTests(unittest.TestCase):
    def test_target_limits(self):
        for spec in ({'target':'hailo8','task':'llm'},{'target':'bogus','task':'vision'},{'target':'hailo8','task':'vision','epochs':0}):
            with self.assertRaises(ValueError):validate(spec)
    def test_artifact_hash_and_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'a';p.write_bytes(b'abc')
            self.assertEqual(artifact(p)['sha256'],'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')
            p.write_bytes(b'')
            with self.assertRaises(ValueError):artifact(p)
    def test_process_must_produce_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'training-result.json').write_text(json.dumps({'metrics':{},'artifact_paths':[str(root/'missing.onnx')]}))
            with patch('pi_trainer.processes.run_process'):
                with self.assertRaises(ValueError):run_training({'task':'vision','target':'hailo8','dataset_dir':tmp},root,lambda _:None,lambda:False)
    def test_default_isolated_worker_and_no_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'model.onnx').write_bytes(b'fixture')
            (root/'training-result.json').write_text(json.dumps({'metrics':{'test_accuracy':.5},'artifact_paths':[str(root/'model.onnx')]}))
            with patch('pi_trainer.processes.run_process') as run:
                result=run_training({'task':'vision','target':'hailo8l','dataset_dir':tmp},root,lambda _:None,lambda:False)
            self.assertEqual(run.call_args.kwargs['env']['HF_HUB_OFFLINE'],'1')
            self.assertEqual(result['artifacts'][0]['bytes'],7)

    def test_relative_local_model_is_resolved_before_changing_directory(self):
        import os
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'base').mkdir();job=root/'job';job.mkdir()
            checkpoint=job/'adapter_model.safetensors';checkpoint.write_bytes(b'fixture')
            (job/'training-result.json').write_text(json.dumps({'metrics':{},'artifact_paths':[str(checkpoint)]}))
            old=Path.cwd()
            try:
                os.chdir(root)
                with patch('pi_trainer.processes.run_process'):
                    result=run_training({'task':'llm','target':'hailo10h','model':'base','dataset_dir':str(root)},job,lambda _:None,lambda:False)
                self.assertEqual(json.loads((job/'training-config.json').read_text())['model'],str((root/'base').resolve()))
                self.assertEqual(result['adapter_path'],str(job.resolve()))
            finally:os.chdir(old)

class ExplicitSplitTests(unittest.TestCase):
    def fixture(self,root):
        for subset in ('train','validation','test'):
            for name in ('a','b'):
                folder=root/subset/name;folder.mkdir(parents=True)
                (folder/'image.png').write_bytes(b'presence-only fixture')
    def test_extra_test_only_class_is_rejected(self):
        from pi_trainer.vision_train_worker import validate_explicit_splits
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root);(root/'test'/'unseen').mkdir()
            (root/'test'/'unseen'/'image.png').write_bytes(b'x')
            with self.assertRaisesRegex(ValueError,'identical class sets'):validate_explicit_splits(root)
    def test_empty_validation_class_is_rejected(self):
        from pi_trainer.vision_train_worker import validate_explicit_splits
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root);(root/'validation'/'b'/'image.png').unlink()
            with self.assertRaisesRegex(ValueError,'validation has no supported images for class b'):validate_explicit_splits(root)
    def test_valid_coverage_passes(self):
        from pi_trainer.vision_train_worker import validate_explicit_splits
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            self.assertEqual(validate_explicit_splits(root),['a','b'])

class OptionalRealTrainingTests(unittest.TestCase):
    def test_tiny_cpu_training_and_onnx_reload(self):
        import importlib.util
        if any(importlib.util.find_spec(name) is None for name in ('torch','onnx','PIL','numpy')):
            self.skipTest('Optional training dependencies not installed')
        import numpy as np
        from PIL import Image
        from onnx.reference import ReferenceEvaluator
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'data'
            for label,channel in [('red',0),('green',1)]:
                (data/label).mkdir(parents=True)
                for i in range(5):
                    pixels=np.zeros((16,16,3),dtype=np.uint8);pixels[:,:,channel]=100+i*20
                    Image.fromarray(pixels).save(data/label/f'{i}.png')
            result=run_training({'task':'vision','target':'hailo8l','dataset_dir':str(data),'python':sys.executable,'image_size':16,'epochs':1},root/'job',lambda _:None,lambda:False)
            self.assertEqual(result['metrics']['train_count'],6)
            self.assertEqual(result['metrics']['validation_count'],2)
            self.assertEqual(result['metrics']['test_count'],2)
            output=ReferenceEvaluator(str(root/'job/model.onnx')).run(None,{'images':np.zeros((1,3,16,16),dtype=np.float32)})[0]
            self.assertEqual(output.shape,(1,2));self.assertTrue(np.isfinite(output).all())
