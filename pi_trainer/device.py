"""Read-only SSH diagnostics using an existing trusted SSH config alias."""
from __future__ import annotations
import argparse
import json
import re
import subprocess
from pathlib import Path


def probe(host: str) -> dict:
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,252}', host):
        raise ValueError('Use an SSH config alias or hostname without spaces/options. Configure user and port in SSH config.')
    source = Path(__file__).with_name('pi_probe.py').read_text()
    result = subprocess.run([
        'ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
        '-o', 'ConnectTimeout=10', host, 'python3', '-'],
        input=source, capture_output=True, text=True, timeout=45)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or 'SSH probe failed')
    report=json.loads(result.stdout)
    if report.get('schema') != 'pi-trainer/probe/v1':
        raise ValueError('Unexpected helper response')
    return {'host':host,'report':report}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['probe'])
    parser.add_argument('host',help='Existing trusted SSH host alias')
    args=parser.parse_args()
    try:print(json.dumps(probe(args.host),indent=2))
    except (ValueError,RuntimeError,OSError,subprocess.TimeoutExpired) as exc:
        parser.exit(1,json.dumps({'error':str(exc)})+'\n')


if __name__ == '__main__':main()
