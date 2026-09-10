import unittest
from unittest.mock import patch
from pi_trainer.device import probe
from pi_trainer.pi_probe import probe as local_probe

class DeviceTests(unittest.TestCase):
    def test_host_injection_rejected_before_ssh(self):
        for value in ['-oProxyCommand=bad', 'pi;touch x', 'pi\nlocalhost', 'user@pi', '']:
            with self.subTest(value=value), self.assertRaises(ValueError):probe(value)
    def test_probe_preserves_host_verification(self):
        import subprocess
        with patch('pi_trainer.device.subprocess.run', return_value=subprocess.CompletedProcess([],0,'{"schema":"pi-trainer/probe/v1"}','')) as run:
            self.assertEqual(probe('lab-pi')['host'],'lab-pi')
            argv=run.call_args.args[0]
            self.assertIn('StrictHostKeyChecking=yes',argv)
            self.assertIn('BatchMode=yes',argv)
            self.assertFalse(run.call_args.kwargs.get('shell',False))
    def test_absent_hailo_never_claims_hardware_success(self):
        with patch('pi_trainer.pi_probe.shutil.which',return_value=None):
            result=local_probe()
        self.assertEqual(result['hailo_identity']['status'],'unavailable')
        self.assertEqual(result['inference_test'],'not_run')
