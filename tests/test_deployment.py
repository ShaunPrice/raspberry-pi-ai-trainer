import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from pi_trainer import deployment, pi_agent


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = self.root / 'model.hef'
        self.model.write_bytes(b'fixture HEF bytes; not a hardware executable')

    def tearDown(self):
        self.temp.cleanup()

    def bundle(self, **extra):
        spec = {'model_path': str(self.model), 'target': 'hailo8l', 'runtime_version': '4.23.0', **extra}
        return deployment.create_bundle(spec, self.root / 'out')

    def rewrite(self, bundle, change):
        path = self.root / 'changed.zip'
        with zipfile.ZipFile(bundle) as src, zipfile.ZipFile(path, 'w') as dst:
            for info in src.infolist():
                name, data = change(info.filename, src.read(info))
                dst.writestr(name, data)
        return path

    def test_valid_streaming_bundle(self):
        scripts = self.root / 'scripts'
        scripts.mkdir()
        (scripts / 'run.py').write_text('print("hi")')
        result = self.bundle(scripts_dir=str(scripts), entrypoint='run.py')
        manifest = deployment.validate_bundle(result['bundle_path'])
        self.assertEqual(manifest['entrypoint'], 'scripts/run.py')
        self.assertEqual(manifest['target'], 'hailo8l')
        self.assertEqual(result['sha256'], pi_agent.digest(result['bundle_path']))

    def test_corruption_rejected(self):
        bundle = self.bundle()['bundle_path']
        path = self.rewrite(bundle, lambda n, d: (n, b'changed' if n.startswith('payload/') else d))
        with self.assertRaises(ValueError):
            deployment.validate_bundle(path)

    def test_path_traversal_rejected(self):
        bundle = self.bundle()['bundle_path']
        path = self.rewrite(bundle, lambda n, d: ('payload/../../outside' if n.startswith('payload/') else n, d))
        with self.assertRaises(ValueError):
            deployment.validate_bundle(path)
        self.assertFalse((self.root / 'outside').exists())

    def test_zip_symlink_rejected(self):
        bundle = self.bundle()['bundle_path']
        with zipfile.ZipFile(bundle, 'a') as z:
            info = zipfile.ZipInfo('payload/link')
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            z.writestr(info, '/etc/passwd')
        with self.assertRaises(ValueError):
            deployment.validate_bundle(bundle)

    def test_duplicate_entry_rejected(self):
        import warnings
        bundle = self.bundle()['bundle_path']
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with zipfile.ZipFile(bundle, 'a') as z:
                z.writestr('manifest.json', '{}')
        with self.assertRaises(ValueError):
            deployment.validate_bundle(bundle)

    def test_wrong_target_or_runtime_rejected(self):
        manifest = self.bundle()['manifest']
        report = {'identity': {'exit_code': 0, 'stdout': 'Device Architecture: HAILO8'},
                  'runtime': {'exit_code': 0, 'stdout': 'HailoRT v4.23.0'}}
        with patch.object(pi_agent, 'probe', return_value=report):
            with self.assertRaisesRegex(ValueError, 'target mismatch'):
                pi_agent.check_device(manifest)
            manifest['target'] = 'hailo8'
            self.assertEqual(pi_agent.check_device(manifest), report)
            manifest['runtime_version'] = '4.22.0'
            with self.assertRaisesRegex(ValueError, 'version'):
                pi_agent.check_device(manifest)

    @unittest.skipIf(os.name == 'nt', 'Pi-side release selection uses POSIX fcntl; Windows host transport is tested separately')
    def test_stage_activate_and_atomic_rollback(self):
        first = self.bundle()
        self.model.write_bytes(b'second fixture')
        second = self.bundle()
        base = self.root / 'agent'
        with patch.object(pi_agent, 'check_device', return_value={'test': 'mock device only'}):
            for bundle in (first, second):
                pi_agent.stage(bundle['bundle_path'], bundle['sha256'], base)
                pi_agent.activate(bundle['manifest']['release_id'], base)
            self.assertEqual(pi_agent.read_state(base)['active'], second['manifest']['release_id'])
            result = pi_agent.rollback(base)
            self.assertEqual(result['active'], first['manifest']['release_id'])
            self.assertEqual(result['previous'], second['manifest']['release_id'])
            self.assertEqual(list(base.glob('.state-*')), [])
            active_payload = base / 'releases' / result['active'] / 'payload/model/model.hef'
            active_payload.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                pi_agent.activate(result['active'], base)

    def test_upload_digest_rejected_before_probe(self):
        b = self.bundle()
        with patch.object(pi_agent, 'check_device') as check:
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                pi_agent.stage(b['bundle_path'], '0' * 64, self.root / 'agent')
            check.assert_not_called()

    def test_host_shell_injection_rejected(self):
        for host in ['-oProxyCommand=evil', 'pi; touch pwn', 'user@pi', 'pi\nwhoami']:
            with self.assertRaises(ValueError):
                deployment._host({'host': host})

    @unittest.skipIf(os.name == 'nt', 'Pi-side release selection uses POSIX fcntl; Windows host transport is tested separately')
    def test_benchmark_is_real_command_and_failure_not_success(self):
        b = self.bundle()
        base = self.root / 'agent'
        with patch.object(pi_agent, 'check_device', return_value={}):
            pi_agent.stage(b['bundle_path'], b['sha256'], base)
            pi_agent.activate(b['manifest']['release_id'], base)
            with patch.object(pi_agent, 'command', return_value={'exit_code': 1, 'stderr': 'invalid HEF'}) as cmd:
                with self.assertRaisesRegex(RuntimeError, 'failed'):
                    pi_agent.benchmark({}, base)
                self.assertEqual(cmd.call_args.args[0][:2], ['hailortcli', 'benchmark'])

    def test_atomic_state_failure_preserves_previous(self):
        base = self.root / 'agent'
        pi_agent.write_state(base, {'active': 'a' * 32, 'previous': None})
        with patch.object(pi_agent.os, 'replace', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                pi_agent.write_state(base, {'active': 'b' * 32, 'previous': 'a' * 32})
        self.assertEqual(pi_agent.read_state(base)['active'], 'a' * 32)
        self.assertEqual(list(base.glob('.state-*')), [])

    @unittest.skipIf(os.name == 'nt', 'Pi-side release selection uses POSIX fcntl; Windows host transport is tested separately')
    def test_real_local_script_execution(self):
        scripts = self.root / 'scripts'
        scripts.mkdir()
        (scripts / 'run.py').write_text('from pathlib import Path; print(Path("model/model.hef").is_file())')
        bundle = self.bundle(scripts_dir=str(scripts), entrypoint='run.py')
        base = self.root / 'agent'
        with patch.object(pi_agent, 'check_device', return_value={}):
            pi_agent.stage(bundle['bundle_path'], bundle['sha256'], base)
            pi_agent.activate(bundle['manifest']['release_id'], base)
            result = pi_agent.benchmark({'mode': 'script'}, base)
        self.assertEqual(result['status'], 'script_completed')
        self.assertEqual(result['result']['stdout'].strip(), 'True')

    def test_direct_llm_requires_explicit_provider(self):
        with self.assertRaisesRegex(ValueError, 'runtime_provider'):
            pi_agent.direct_llm(self.bundle()['manifest'], self.root, {})

    def test_direct_llm_api_contract_with_mock_runtime(self):
        import types
        from unittest.mock import MagicMock
        device_class = MagicMock()
        model_class = MagicMock()
        model_class.return_value.generate_all.return_value = 'fixture generated text'
        fake = types.ModuleType('hailo_platform')
        fake.VDevice = device_class
        genai = types.ModuleType('hailo_platform.genai')
        genai.LLM = model_class
        bundle = self.bundle(target='hailo10h', runtime_version='5.2.0', runtime_provider='hailort-genai-llm-v1')
        with patch.dict(sys.modules, {'hailo_platform': fake, 'hailo_platform.genai': genai}):
            result = pi_agent.direct_llm(bundle['manifest'], self.root, {'prompt': 'hello', 'max_tokens': 20})
        self.assertEqual(result['status'], 'inference_completed')
        self.assertEqual(result['response'], 'fixture generated text')
        self.assertEqual(model_class.return_value.generate_all.call_args.kwargs['max_generated_tokens'], 20)
        model_class.return_value.release.assert_called_once()
        device_class.return_value.release.assert_called_once()

    @unittest.skipIf(os.name == 'nt', 'Pi-side release selection uses POSIX fcntl; Windows host transport is tested separately')
    def test_hailo10h_vision_uses_real_benchmark_command(self):
        bundle = self.bundle(target='hailo10h', runtime_version='5.2.0')
        base = self.root / 'agent'
        with patch.object(pi_agent, 'check_device', return_value={}):
            pi_agent.stage(bundle['bundle_path'], bundle['sha256'], base)
            pi_agent.activate(bundle['manifest']['release_id'], base)
            with patch.object(pi_agent, 'command', return_value={'exit_code': 0, 'stdout': 'mock only', 'stderr': ''}) as cmd:
                result = pi_agent.benchmark({'mode': 'vision'}, base)
        self.assertEqual(result['status'], 'benchmark_completed')
        self.assertEqual(cmd.call_args.args[0][:2], ['hailortcli', 'benchmark'])

    @unittest.skipIf(os.name == 'nt', 'Pi-side release selection uses POSIX fcntl; Windows host transport is tested separately')
    def test_genai_provider_rejects_vision_benchmark(self):
        bundle = self.bundle(target='hailo10h', runtime_version='5.2.0', runtime_provider='hailort-genai-llm-v1')
        base = self.root / 'agent'
        with patch.object(pi_agent, 'check_device', return_value={}):
            pi_agent.stage(bundle['bundle_path'], bundle['sha256'], base)
            pi_agent.activate(bundle['manifest']['release_id'], base)
            with self.assertRaisesRegex(ValueError, 'GenAI bundles'):
                pi_agent.benchmark({'mode': 'vision'}, base)

    def test_transport_cancellation_preserves_job_cancelled_type(self):
        from pi_trainer.processes import JobCancelled
        count = 0
        def cancel():
            nonlocal count
            count += 1
            return count >= 3
        with self.assertRaisesRegex(JobCancelled, 'remote operation state may be unknown'):
            deployment._run([sys.executable, '-c', 'import time; time.sleep(30)'], cancel)

    def test_transport_timeout_reports_unknown_remote_state(self):
        with self.assertRaisesRegex(TimeoutError, 'remote operation state may be unknown'):
            deployment._run([sys.executable, '-c', 'import time; time.sleep(30)'], lambda: False, timeout=0.05)

    def test_helper_local_status_smoke(self):
        # This checks the actual standalone command path; it never touches Hailo hardware.
        import os
        env = {**os.environ, 'HOME': str(self.root), 'USERPROFILE': str(self.root), 'PYTHONPYCACHEPREFIX': str(self.root / 'cache')}
        result = subprocess.run([sys.executable, str(Path(pi_agent.__file__)), '{"action":"status"}'],
                                capture_output=True, text=True, env=env, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['result'], {'active': None, 'previous': None})

    def test_cancel_does_not_publish_bundle(self):
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            deployment.create_bundle({'model_path': str(self.model), 'target': 'hailo8l', 'runtime_version': '4.23.0'},
                                     self.root / 'out', cancelled=lambda: True)
        self.assertEqual(list((self.root / 'out').iterdir()), [])


if __name__ == '__main__':
    unittest.main()
