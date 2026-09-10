#!/usr/bin/env python3
"""Read-only Pi/Hailo diagnostics. No package changes or inference are performed."""
import json
import platform
import shutil
import subprocess
from pathlib import Path


def command(argv):
    if not shutil.which(argv[0]):
        return {'status': 'unavailable', 'reason': argv[0] + ' not installed'}
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=15)
        return {'status': 'ok' if p.returncode == 0 else 'error',
                'exit_code': p.returncode, 'stdout': p.stdout[:32768],
                'stderr': p.stderr[:4096]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'status': 'error', 'reason': str(exc)}


def probe():
    os_release = Path('/etc/os-release')
    model = Path('/proc/device-tree/model')
    return {'schema': 'pi-trainer/probe/v1',
            'machine': platform.machine(), 'system': platform.system(),
            'kernel': platform.release(),
            'os_release': os_release.read_text() if os_release.exists() else None,
            'board': model.read_text().strip('\0') if model.exists() else None,
            'hailo_identity': command(['hailortcli', 'fw-control', 'identify']),
            'hailo_runtime': command(['hailortcli', '--version']),
            'disk_free_bytes': shutil.disk_usage(Path.home()).free,
            'inference_test': 'not_run'}


if __name__ == '__main__':
    print(json.dumps(probe(), indent=2))
