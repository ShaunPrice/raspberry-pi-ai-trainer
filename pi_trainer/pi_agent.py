#!/usr/bin/env python3
"""Unprivileged standalone Raspberry Pi deployment agent (Python standard library)."""
import hashlib
from contextlib import contextmanager
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

SCHEMA = 'pi-trainer/deployment/v1'
MAX_FILE = 16 * 1024**3
MAX_TOTAL = 32 * 1024**3
TARGETS = {'hailo8l', 'hailo8', 'hailo10h'}


def safe_path(value):
    if not isinstance(value, str) or not value or '\\' in value or '\x00' in value:
        raise ValueError('Invalid payload path')
    p = PurePosixPath(value)
    if p.is_absolute() or any(x in ('', '.', '..') for x in value.split('/')) or ':' in value:
        raise ValueError('Unsafe payload path: ' + value)
    return value


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_bundle(path):
    with zipfile.ZipFile(path) as z:
        entries = z.infolist()
        names = [i.filename for i in entries]
        if len(names) != len(set(names)) or len(names) > 10000:
            raise ValueError('Duplicate entries or excessive bundle files')
        if 'manifest.json' not in names or z.getinfo('manifest.json').file_size > 1024 * 1024:
            raise ValueError('Missing or oversized manifest')
        total = 0
        for info in entries:
            safe_path(info.filename)
            mode = info.external_attr >> 16
            if info.is_dir() or stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG)):
                raise ValueError('Only regular payload files allowed')
            total += info.file_size
            if info.file_size > MAX_FILE or total > MAX_TOTAL:
                raise ValueError('Bundle size limit exceeded')
        m = json.loads(z.read('manifest.json'))
        if m.get('schema') != SCHEMA or m.get('target') not in TARGETS:
            raise ValueError('Unsupported bundle schema or target')
        if not re.fullmatch(r'[a-f0-9]{16,64}', m.get('release_id', '')):
            raise ValueError('Invalid release ID')
        if not re.fullmatch(r'\d+\.\d+\.\d+', m.get('runtime_version', '')):
            raise ValueError('runtime_version must be exact major.minor.patch')
        records = m.get('files')
        if not isinstance(records, list) or not records:
            raise ValueError('Empty manifest')
        expected = {'manifest.json'}
        for record in records:
            name = 'payload/' + safe_path(record['path'])
            if name in expected:
                raise ValueError('Duplicate manifest path')
            expected.add(name)
            if name not in names or record['bytes'] != z.getinfo(name).file_size:
                raise ValueError('Payload size mismatch')
            h = hashlib.sha256()
            actual_size = 0
            with z.open(name) as src:
                for block in iter(lambda: src.read(1024 * 1024), b''):
                    actual_size += len(block)
                    if actual_size > MAX_FILE:
                        raise ValueError('Expanded file exceeds size limit')
                    h.update(block)
            if h.hexdigest() != record['sha256']:
                raise ValueError('Payload checksum mismatch')
        if set(names) != expected:
            raise ValueError('Unlisted bundle entries')
        for key in ('model_path', 'entrypoint'):
            if m.get(key) is not None and 'payload/' + safe_path(m[key]) not in expected:
                raise ValueError('Missing ' + key)
        if m.get('runtime_provider') not in (None, 'hailort-genai-llm-v1'):
            raise ValueError('Unsupported runtime_provider')
        if m.get('runtime_provider') == 'hailort-genai-llm-v1' and m['target'] != 'hailo10h':
            raise ValueError('GenAI LLM provider requires Hailo-10H')
        if not str(m.get('model_path', '')).lower().endswith('.hef'):
            raise ValueError('Compiled .hef model required')
        return m


def command(argv, timeout=30, cwd=None):
    import signal
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        try:
            process = subprocess.Popen(argv, stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                                       cwd=cwd, start_new_session=True)
        except OSError as exc:
            return {'exit_code': -1, 'stderr': str(exc), 'stdout': ''}
        started = time.monotonic()
        reason = None
        try:
            while process.poll() is None:
                if time.monotonic() - started > timeout:
                    reason = 'Device command timed out'
                    break
                if os.fstat(out.fileno()).st_size + os.fstat(err.fileno()).st_size > 32 * 1024 * 1024:
                    reason = 'Device command output exceeded 32 MiB'
                    break
                time.sleep(0.1)
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                except ProcessLookupError:
                    process.wait()
        out.seek(max(0, os.fstat(out.fileno()).st_size - 32768))
        err.seek(max(0, os.fstat(err.fileno()).st_size - 8192))
        return {'exit_code': -1 if reason else process.returncode,
                'stdout': out.read().decode(errors='replace'),
                'stderr': reason or err.read().decode(errors='replace')}


def probe():
    return {'schema': 'pi-trainer/device/v1', 'identity': command(['hailortcli', 'fw-control', 'identify']),
            'runtime': command(['hailortcli', '--version'])}


def check_device(manifest):
    report = probe()
    if any(report[k]['exit_code'] for k in ('identity', 'runtime')):
        raise RuntimeError('Hailo device/runtime probe failed: ' + json.dumps(report))
    identities = re.findall(r'HAILO[ _-]?(8L|10H|8)(?![A-Za-z0-9])', report['identity']['stdout'], re.I)
    detected = {'hailo' + x.lower() for x in identities}
    if manifest['target'] not in detected:
        raise ValueError('Device target mismatch: ' + str(sorted(detected)))
    versions = re.findall(r'(?<![0-9])\d+\.\d+\.\d+(?![0-9])', report['runtime']['stdout'])
    if manifest['runtime_version'] not in versions:
        raise ValueError('Installed HailoRT version does not match bundle runtime_version')
    return report


def base_dir():
    return Path.home() / '.local/share/pi-trainer'


def release_dir(base, release_id):
    if not re.fullmatch(r'[a-f0-9]{16,64}', release_id):
        raise ValueError('Invalid release ID')
    return base / 'releases' / release_id


def read_state(base):
    p = base / 'state.json'
    return json.loads(p.read_text()) if p.exists() else {'active': None, 'previous': None}


def write_state(base, state):
    base.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.state-', dir=base)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, base / 'state.json')
    finally:
        if os.path.exists(name):
            os.unlink(name)


def verify_release(path):
    m = json.loads((path / 'manifest.json').read_text())
    for f in m['files']:
        p = path / 'payload' / safe_path(f['path'])
        if p.is_symlink() or not p.is_file() or p.stat().st_size != f['bytes'] or digest(p) != f['sha256']:
            raise ValueError('Staged release checksum mismatch')
    return m


def stage(bundle, expected_sha256, base=None):
    base = Path(base) if base is not None else base_dir()
    if digest(bundle) != expected_sha256:
        raise ValueError('Uploaded bundle SHA256 mismatch')
    m = validate_bundle(bundle)
    report = check_device(m)
    destination = release_dir(base, m['release_id'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if verify_release(destination) != m:
            raise ValueError('Release ID collision')
    else:
        temp = Path(tempfile.mkdtemp(prefix='.stage-', dir=destination.parent))
        try:
            with zipfile.ZipFile(bundle) as z:
                for info in z.infolist():
                    out = temp / info.filename
                    out.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(info) as src, out.open('wb') as dst:
                        shutil.copyfileobj(src, dst, 1024 * 1024)
            os.replace(temp, destination)
        finally:
            if temp.exists():
                shutil.rmtree(temp)
    return {'release_id': m['release_id'], 'status': 'staged', 'device': report}


@contextmanager
def state_lock(base):
    import fcntl
    base.mkdir(parents=True, exist_ok=True)
    with (base / '.state.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _activate(release_id, base=None):
    base = Path(base) if base is not None else base_dir()
    manifest = verify_release(release_dir(base, release_id))
    check_device(manifest)
    state = read_state(base)
    if state['active'] != release_id:
        write_state(base, {'active': release_id, 'previous': state['active']})
    return {'status': 'active', **read_state(base), 'note': 'Release selected; services are not installed or started.'}


def activate(release_id, base=None):
    base = Path(base) if base is not None else base_dir()
    with state_lock(base):
        return _activate(release_id, base)


def rollback(base=None):
    base = Path(base) if base is not None else base_dir()
    with state_lock(base):
        state = read_state(base)
        if not state['previous']:
            raise ValueError('No previous release')
        return _activate(state['previous'], base)


def direct_llm(manifest, release, request):
    if manifest.get('runtime_provider') != 'hailort-genai-llm-v1' or manifest['target'] != 'hailo10h':
        raise ValueError('Direct LLM testing requires a Hailo-10H bundle declaring runtime_provider=hailort-genai-llm-v1')
    try:
        from hailo_platform import VDevice
        from hailo_platform.genai import LLM
    except ImportError as exc:
        raise RuntimeError('Install the qualified HailoRT GenAI Python runtime on the Pi before direct LLM testing') from exc
    max_tokens = request.get('max_tokens', 200)
    if type(max_tokens) is not int or not 1 <= max_tokens <= 2048:
        raise ValueError('max_tokens must be an integer between 1 and 2048')
    text = request.get('prompt', 'Reply with hello.')
    if not isinstance(text, str) or not text or len(text) > 8192:
        raise ValueError('Prompt must contain 1 to 8192 characters')
    device = None
    llm = None
    started = time.monotonic()
    try:
        params = VDevice.create_params()
        params.group_id = 'SHARED'
        device = VDevice(params)
        # GenAI-compatible HEFs embed their model/tokenizer configuration. Ordinary vision
        # HEFs, GGUF files, and raw checkpoints cannot be registered by this constructor.
        llm = LLM(device, str(release / 'payload' / manifest['model_path']))
        messages = [{'role': 'user', 'content': [{'type': 'text', 'text': text}]}]
        response = llm.generate_all(prompt=messages, temperature=0.1, seed=42, max_generated_tokens=max_tokens)
        if not isinstance(response, str) or not response.strip():
            raise RuntimeError('GenAI runtime returned no text')
        return {'status': 'inference_completed', 'release_id': manifest['release_id'],
                'provider': 'hailort-genai-llm-v1', 'response': response[:1024 * 1024],
                'elapsed_seconds': time.monotonic() - started,
                'note': 'Direct HailoRT GenAI inference from selected HEF; successful generation does not measure task accuracy.'}
    finally:
        if llm is not None:
            try:
                llm.clear_context()
            finally:
                try:
                    llm.release()
                finally:
                    if device is not None:
                        device.release()
        elif device is not None:
            device.release()


def benchmark(request, base=None):
    base = Path(base) if base is not None else base_dir()
    if request.get('mode') == 'llm':
        device = probe()
        if device['identity']['exit_code'] or not re.search(r'HAILO[ _-]?10H(?![A-Za-z0-9])', device['identity']['stdout'], re.I):
            raise ValueError('LLM testing requires a detected Hailo-10H device')
        name = request.get('model')
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_.:/-]{1,200}', name):
            raise ValueError('Explicit registered hailo-ollama model name required')
        data = json.dumps({'model': name, 'prompt': str(request.get('prompt', 'Reply with hello.'))[:8192], 'stream': False}).encode()
        started = time.monotonic()
        req = urllib.request.Request('http://127.0.0.1:8000/api/generate', data=data, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=180) as response:
            raw = response.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                raise ValueError('Oversized LLM response')
            result = json.loads(raw)
        if result.get('error') or not result.get('response'):
            raise RuntimeError('LLM inference failed: ' + json.dumps(result))
        return {'status': 'inference_completed', 'elapsed_seconds': time.monotonic()-started, 'result': result,
                'note': 'Uses an already registered hailo-ollama model; HEF registration is not automatic.'}
    state = read_state(base)
    if not state['active']:
        raise ValueError('No active release')
    release = release_dir(base, state['active'])
    m = verify_release(release)
    check_device(m)
    if request.get('mode') == 'llm_direct':
        return direct_llm(m, release, request)
    if request.get('mode') == 'script':
        entrypoint = m.get('entrypoint')
        if not entrypoint:
            raise ValueError('Active release has no entrypoint')
        interpreter = {'.py': 'python3', '.sh': '/bin/sh'}.get(Path(entrypoint).suffix)
        if not interpreter:
            raise ValueError('Script test supports .py and .sh entrypoints only')
        result = command([interpreter, str(release / 'payload' / entrypoint)], timeout=180, cwd=str(release / 'payload'))
        if result['exit_code']:
            raise RuntimeError('Release script failed: ' + json.dumps(result))
        return {'status': 'script_completed', 'release_id': state['active'], 'result': result,
                'note': 'Entrypoint exited successfully; inspect its output for application accuracy.'}
    if request.get('mode') not in (None, 'vision'):
        raise ValueError('Unknown test mode')
    if m.get('runtime_provider') is not None:
        raise ValueError('GenAI bundles require llm_direct or script testing; vision benchmarking does not execute the GenAI provider')
    result = command(['hailortcli', 'benchmark', str(release / 'payload' / m['model_path'])], timeout=180)
    if result['exit_code']:
        raise RuntimeError('Hailo benchmark failed: ' + json.dumps(result))
    return {'status': 'benchmark_completed', 'release_id': state['active'], 'result': result,
            'note': 'Synthetic HailoRT benchmark; model accuracy is not evaluated.'}


def dispatch(request):
    action = request['action']
    if action == 'probe':
        return probe()
    if action == 'status':
        return read_state(base_dir())
    if action == 'prepare':
        token = request['token']
        if not re.fullmatch(r'[a-f0-9]{32}', token):
            raise ValueError('Invalid upload token')
        folder = base_dir() / 'incoming'
        folder.mkdir(parents=True, exist_ok=True)
        return {'path': str(folder / (token + '.zip'))}
    if action == 'stage':
        token = request['token']
        if not re.fullmatch(r'[a-f0-9]{32}', token):
            raise ValueError('Invalid upload token')
        bundle = base_dir() / 'incoming' / (token + '.zip')
        try:
            return stage(bundle, request['sha256'])
        finally:
            bundle.unlink(missing_ok=True)
    if action == 'activate':
        return activate(request['release_id'])
    if action == 'rollback':
        return rollback()
    if action == 'benchmark':
        return benchmark(request)
    raise ValueError('Unknown agent action')


if __name__ == '__main__':
    try:
        print(json.dumps({'ok': True, 'result': dispatch(json.loads(sys.argv[1]))}))
    except Exception as exc:
        print(json.dumps({'ok': False, 'error': str(exc)}))
        sys.exit(1)
