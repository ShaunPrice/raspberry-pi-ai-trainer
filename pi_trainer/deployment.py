"""Streaming deployment packages and trusted SSH transport to the Pi agent."""
from __future__ import annotations
import base64
import json
import os
import re
import subprocess
import uuid
import zipfile
from pathlib import Path
from .pi_agent import SCHEMA, TARGETS, digest, safe_path, validate_bundle
from .processes import JobCancelled, stop_process


def _check_cancel(cancelled):
    if cancelled():
        raise JobCancelled('Operation cancelled')


def create_bundle(spec: dict, workdir: Path, emit=lambda value: None, cancelled=lambda: False) -> dict:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    model = Path(spec['model_path']).expanduser().resolve(strict=True)
    if not model.is_file() or model.suffix.lower() != '.hef':
        raise ValueError('A compiled Hailo .hef model is required')
    if spec.get('target') not in TARGETS:
        raise ValueError('Unsupported target')
    if not re.fullmatch(r'\d+\.\d+\.\d+', spec.get('runtime_version', '')):
        raise ValueError('Specify exact installed HailoRT runtime_version major.minor.patch')
    sources = {'model/' + model.name: model}
    for kind, key in [('scripts', 'scripts_dir'), ('assets', 'assets_dir')]:
        if spec.get(key):
            folder = Path(spec[key]).expanduser().resolve(strict=True)
            if not folder.is_dir():
                raise ValueError(key + ' must be a directory')
            for path in sorted(folder.rglob('*')):
                if path.is_symlink():
                    raise ValueError('Symlinks are not allowed in bundles')
                if path.is_file():
                    sources[kind + '/' + path.relative_to(folder).as_posix()] = path
    entrypoint = spec.get('entrypoint') or None
    if entrypoint is not None:
        safe_path(entrypoint)
        if not entrypoint.startswith('scripts/'):
            entrypoint = 'scripts/' + entrypoint
        if entrypoint not in sources:
            raise ValueError('Entrypoint must name a bundled script')
    manifest = {'schema': SCHEMA, 'release_id': uuid.uuid4().hex, 'target': spec['target'],
                'runtime_version': spec['runtime_version'], 'model_path': 'model/' + model.name,
                'entrypoint': entrypoint, 'files': []}
    if spec.get('runtime_provider'):
        manifest['runtime_provider'] = spec['runtime_provider']
    output = workdir / (manifest['release_id'] + '.zip')
    partial = output.with_suffix('.partial')
    try:
        with zipfile.ZipFile(partial, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for name, source in sources.items():
                _check_cancel(cancelled)
                safe_path(name)
                # Hash the bytes written, avoiding hash/copy races on changing inputs.
                import hashlib
                h = hashlib.sha256()
                size = 0
                with source.open('rb') as src, archive.open('payload/' + name, 'w', force_zip64=True) as dst:
                    for block in iter(lambda: src.read(1024 * 1024), b''):
                        _check_cancel(cancelled)
                        dst.write(block)
                        h.update(block)
                        size += len(block)
                manifest['files'].append({'path': name, 'sha256': h.hexdigest(), 'bytes': size})
                emit('Bundled ' + name)
            archive.writestr('manifest.json', json.dumps(manifest, indent=2))
        validate_bundle(partial)
        partial.replace(output)
    finally:
        partial.unlink(missing_ok=True)
    return {'status': 'bundle_created', 'bundle_path': str(output), 'sha256': digest(output), 'manifest': manifest}


def _host(spec):
    host = spec.get('host', '')
    if not isinstance(host, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,252}', host):
        raise ValueError('Use a trusted SSH config host alias; user/port belong in SSH config')
    return host


def _run(argv, cancelled, timeout=300, input_text=None):
    import time
    import tempfile
    _check_cancel(cancelled)
    # Disk-backed output prevents SSH errors or device programs consuming host RAM.
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err, tempfile.TemporaryFile() as inp:
        if input_text is not None:
            inp.write(input_text.encode())
            inp.seek(0)
        kwargs = {'start_new_session': True} if os.name == 'posix' else {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP}
        p = subprocess.Popen(argv, stdin=inp, stdout=out, stderr=err, **kwargs)
        started = time.monotonic()
        try:
            while p.poll() is None:
                _check_cancel(cancelled)
                if time.monotonic() - started > timeout:
                    raise TimeoutError('SSH operation timed out; remote operation state may be unknown. Terminating SSH does not prove the remote operation stopped; inspect the Pi before retrying.')
                time.sleep(0.1)
        except JobCancelled as exc:
            raise JobCancelled('SSH operation cancelled; remote operation state may be unknown. Terminating SSH does not prove the remote operation stopped; inspect the Pi before retrying.') from exc
        finally:
            if p.poll() is None:
                stop_process(p)
        out.seek(0, 2)
        if out.tell() > 1024 * 1024:
            raise ValueError('Oversized agent response')
        out.seek(0)
        err.seek(0)
        stdout = out.read().decode(errors='replace')
        stderr = err.read(16384).decode(errors='replace')
        if p.returncode:
            raise RuntimeError('Remote command failed; remote operation state may be unknown. Inspect the Pi before retrying: ' + (stderr or stdout))
        return stdout


def _agent(host, request, cancelled):
    source = Path(__file__).with_name('pi_agent.py').read_text()
    encoded = base64.b64encode(json.dumps(request).encode()).decode()
    # The command string is constant; request content travels only on stdin.
    source = source.replace("if __name__ == '__main__':", "if False:")
    source += '\nimport base64\nprint(json.dumps(dispatch(json.loads(base64.b64decode("' + encoded + '")))))\n'
    output = _run(['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o',
                   'ConnectTimeout=10', host, 'python3 -'], cancelled, input_text=source)
    return json.loads(output)


def deploy(spec: dict, workdir: Path, emit=lambda value: None, cancelled=lambda: False) -> dict:
    host = _host(spec)
    bundle = Path(spec['bundle_path']).expanduser().resolve(strict=True)
    manifest = validate_bundle(bundle)
    token = uuid.uuid4().hex
    emit('Preparing trusted SSH upload')
    _agent(host, {'action': 'prepare', 'token': token}, cancelled)
    # Relative SFTP path is rooted at the remote user's home, never a user supplied shell path.
    remote = host + ':.local/share/pi-trainer/incoming/' + token + '.zip'
    _run(['scp', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=10',
          str(bundle), remote], cancelled, timeout=3600)
    emit('Verifying uploaded model and Hailo runtime')
    staged = _agent(host, {'action': 'stage', 'token': token, 'sha256': digest(bundle)}, cancelled)
    if spec.get('activate', True):
        result = _agent(host, {'action': 'activate', 'release_id': manifest['release_id']}, cancelled)
    else:
        result = staged
    return {'host': host, **result}


def test_device(spec: dict, workdir: Path, emit=lambda value: None, cancelled=lambda: False) -> dict:
    host = _host(spec)
    action = spec.get('action', 'benchmark')
    if action not in ('probe', 'status', 'benchmark'):
        raise ValueError('Test action must be probe, status or benchmark')
    emit('Running Pi ' + action)
    request = {key: spec[key] for key in ('mode', 'model', 'prompt', 'max_tokens') if key in spec}
    request['action'] = action
    return {'host': host, **_agent(host, request, cancelled)}


def rollback(spec: dict, workdir: Path, emit=lambda value: None, cancelled=lambda: False) -> dict:
    host = _host(spec)
    emit('Restoring previous verified release')
    return {'host': host, **_agent(host, {'action': 'rollback'}, cancelled)}
