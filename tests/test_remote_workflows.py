import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from pi_trainer.workflows import execute
from pi_trainer.remote import normalise_downloaded_result,run_remote
from pi_trainer.remote_worker import run

class RemoteWorkflowTests(unittest.TestCase):
    def test_optimise_worker_routes_after_materialising(self):
        with tempfile.TemporaryDirectory() as tmp:
            data={key:str(Path(tmp)/key) for key in ('dataset_dir','data','validation_data','test_data')}
            job={'kind':'optimise','project_id':'p','spec':{'worker_id':'w'}}
            with patch('pi_trainer.workflows.materialise',return_value=data) as materialise,patch('pi_trainer.workflows.settings_get',return_value=[{'id':'w'}]),patch('pi_trainer.remote.run_remote',return_value={'remote':True}) as remote,patch('pi_trainer.optimisation.optimise') as local:
                result=execute(object(),job,Path(tmp),lambda _:None,lambda:False)
                self.assertEqual(result,{'remote':True});local.assert_not_called();materialise.assert_called_once()
                self.assertEqual(remote.call_args.args[1],'optimise')
                self.assertEqual(remote.call_args.args[2]['dataset_dir'],data['dataset_dir'])
    def test_sdk_python_is_separate_from_worker_controller(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            worker={'host':'trusted-worker','python':'python3.11','name':'Linux'}
            with patch('pi_trainer.remote.run_process',side_effect=RuntimeError('stop before network')):
                with self.assertRaisesRegex(RuntimeError,'stop before network'):
                    run_remote(worker,'compile',{'provider_python':'/opt/hailo-sdk/bin/python'},root,lambda _:None,lambda:False)
            with zipfile.ZipFile(root/'remote-input.zip') as z:request=json.loads(z.read('request.json'))
            self.assertEqual(request['spec']['python'],'/opt/hailo-sdk/bin/python')
            self.assertNotIn('provider_python',request['spec'])

    def test_unsupported_remote_kind_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'Remote worker supports'):
                run_remote({},'package',{},Path(tmp),lambda _:None,lambda:False)
    def test_downloaded_paths_only_promoted_from_verified_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();output=root/'outputs';(output/'adapter').mkdir(parents=True)
            names=['model.onnx','calibration.npy','adapter/adapter_model.safetensors']
            checks={}
            for name in names:
                p=output/name;p.write_bytes(b'fixture');checks['outputs/'+name]=hashlib.sha256(b'fixture').hexdigest()
            original={'model_path':'/remote/job/outputs/model.onnx','calibration_path':'/remote/job/outputs/calibration.npy','adapter_path':'/remote/job/outputs/adapter','image_path':'/remote/job/outputs/missing.img','bundle_path':'/remote/private/bundle.zip','artifacts':[{'path':'/remote/job/outputs/model.onnx'}],'trials':[{'result':{'model_path':'/remote/job/outputs/model.onnx'}}]}
            report={'output_root':'/remote/job/outputs','files':checks,'result':original}
            result=normalise_downloaded_result(report,root)
            self.assertEqual(result['model_path'],str(output/'model.onnx'))
            self.assertEqual(result['adapter_path'],str(output/'adapter'))
            self.assertNotIn('image_path',result);self.assertNotIn('bundle_path',result)
            self.assertEqual(result['trials'][0]['result']['model_path'],str(output/'model.onnx'))
            self.assertEqual(original['model_path'],'/remote/job/outputs/model.onnx')
    def test_worker_executes_optimisation_and_records_output_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);(folder/'request.json').write_text(json.dumps({'kind':'optimise','spec':{},'path_fields':{}}))
            def optimise(spec,output,emit,cancelled):
                (output/'model.onnx').write_bytes(b'fixture')
                return {'model_path':str(output/'model.onnx')}
            with patch('pi_trainer.remote_worker.platform.system',return_value='Linux'),patch('pi_trainer.optimisation.optimise',side_effect=optimise) as provider:
                run(folder)
            provider.assert_called_once()
            with zipfile.ZipFile(folder/'output.zip') as archive:report=json.loads(archive.read('result.json'))
            self.assertEqual(report['status'],'succeeded')
            self.assertEqual(report['output_root'],str((folder/'outputs').resolve()))
            self.assertIn('outputs/model.onnx',report['files'])
