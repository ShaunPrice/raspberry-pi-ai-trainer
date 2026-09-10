"""Customise a copy of an existing partitioned Raspberry Pi OS image.

No block devices, mounts or elevated host privileges are used. ext4 work is done
on an extracted regular file, then copied back only after filesystem validation.
"""
from __future__ import annotations

import hashlib
import json
import lzma
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import struct
import subprocess
import sys
import uuid
import zipfile
import zlib

LINUX_GUID = bytes.fromhex('af3dc60f838472478e793d69d8477de4')
MAX_BUNDLE_BYTES = 32 * 1024**3


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def regular_file(value):
    path = Path(value).expanduser().resolve()
    if not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
        raise ValueError('Expected a regular file: ' + str(path))
    return path


def inspect_partitions(path):
    """Read 512-byte-sector MBR or CRC-checked GPT; reject overlaps/bounds errors."""
    path = regular_file(path)
    size = path.stat().st_size
    with path.open('rb') as f:
        mbr = f.read(512)
        if len(mbr) != 512 or mbr[510:] != b'\x55\xaa':
            raise ValueError('Image has no valid MBR partition table')
        entries = [mbr[446+i*16:462+i*16] for i in range(4)]
        result = []
        if any(e[4] == 0xee for e in entries):
            header = f.read(512)
            if header[:8] != b'EFI PART':
                raise ValueError('Protective MBR has no GPT header')
            length, expected = struct.unpack_from('<II', header, 12)
            if not 92 <= length <= 512:
                raise ValueError('Invalid GPT header length')
            checked = bytearray(header[:length]); checked[16:20] = b'\0'*4
            if zlib.crc32(checked) & 0xffffffff != expected:
                raise ValueError('GPT header CRC mismatch')
            first, last = struct.unpack_from('<QQ', header, 40)
            table_lba, count, entry_size, crc = struct.unpack_from('<QIII', header, 72)
            if count > 4096 or entry_size < 128 or entry_size > 4096 or table_lba*512+count*entry_size > size:
                raise ValueError('Invalid GPT partition array')
            f.seek(table_lba*512); table = f.read(count*entry_size)
            if zlib.crc32(table) & 0xffffffff != crc:
                raise ValueError('GPT partition array CRC mismatch')
            for i in range(count):
                e = table[i*entry_size:(i+1)*entry_size]
                if e[:16] == b'\0'*16: continue
                start, end = struct.unpack_from('<QQ', e, 32)
                if start < first or end > last or end < start:
                    raise ValueError('GPT partition outside usable sectors')
                result.append(dict(number=i+1, offset=start*512, size=(end-start+1)*512, linux=e[:16]==LINUX_GUID))
        else:
            for i, e in enumerate(entries):
                start, count = struct.unpack_from('<II', e, 8)
                if not e[4]: continue
                if e[4] in (5, 15, 0x85):
                    raise ValueError('Extended MBR partitions are unsupported')
                result.append(dict(number=i+1, offset=start*512, size=count*512, linux=e[4]==0x83))
        previous_end = 512
        for p in sorted(result, key=lambda p:p['offset']):
            if p['offset'] < previous_end or p['size'] < 2048 or p['offset']+p['size'] > size:
                raise ValueError('Partition overlap or image boundary violation')
            previous_end = p['offset']+p['size']
            f.seek(p['offset']+1080)
            p['ext4'] = f.read(2) == b'\x53\xef'
    return result


def select_partition(partitions, number=None):
    matches = [p for p in partitions if p['linux'] and p['ext4'] and (number is None or p['number']==int(number))]
    if len(matches) != 1:
        raise ValueError('Specify a unique Linux ext4 root partition number; found ' + str(len(matches)))
    return matches[0]


def safe_path(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_.\-/]+', value):
        raise ValueError('Bundle paths require safe portable filename characters')
    p = PurePosixPath(value)
    if p.is_absolute() or any(x in ('', '.', '..') for x in value.split('/')):
        raise ValueError('Unsafe bundle path')
    return value


def unpack_bundle(path, destination):
    """Validate the entire archive and hashes before writing any payload files."""
    with zipfile.ZipFile(regular_file(path)) as z:
        infos = z.infolist()
        names = [i.filename for i in infos]
        if len(names) > 20000 or len(set(names)) != len(names):
            raise ValueError('Duplicate archive members or excessive file count')
        if sum(i.file_size for i in infos) > MAX_BUNDLE_BYTES:
            raise ValueError('Bundle exceeds expanded size limit')
        for i in infos:
            safe_path(i.filename)
            mode = i.external_attr >> 16
            if stat.S_IFMT(mode) not in (0, stat.S_IFREG) or i.is_dir() or i.flag_bits & 1:
                raise ValueError('Bundle must contain unencrypted regular files')
            if i.file_size > 1024**2 and i.file_size > max(i.compress_size,1)*1000:
                raise ValueError('Suspicious archive expansion ratio')
        if 'manifest.json' not in names or z.getinfo('manifest.json').file_size > 1024**2:
            raise ValueError('Missing or excessive deployment manifest')
        manifest = json.loads(z.read('manifest.json'))
        if manifest.get('schema') != 'pi-trainer/deployment/v1':
            raise ValueError('Expected deployment bundle schema pi-trainer/deployment/v1')
        release = manifest.get('release_id', '')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,99}', release):
            raise ValueError('Unsafe release ID')
        files = manifest.get('files')
        if not isinstance(files,list) or not files:
            raise ValueError('Bundle has no files')
        declared = {}
        for item in files:
            name = safe_path(item['path'])
            # Manifest paths are relative to payload, with payload/ accepted for compatibility.
            relative = name.removeprefix('payload/')
            safe_path(relative)
            if relative == 'manifest.json': raise ValueError('Reserved payload manifest filename')
            archive = 'payload/' + relative
            if archive in declared: raise ValueError('Duplicate manifest file')
            if type(item.get('bytes')) is not int or item['bytes'] < 0:
                raise ValueError('Invalid declared file size')
            if archive not in names or z.getinfo(archive).file_size != item['bytes']:
                raise ValueError('Bundle file size mismatch')
            h = hashlib.sha256()
            with z.open(archive) as f:
                for block in iter(lambda:f.read(1024**2), b''): h.update(block)
            if h.hexdigest() != item.get('sha256'): raise ValueError('Bundle file hash mismatch')
            declared[archive] = relative
        if set(names) != {'manifest.json', *declared}:
            raise ValueError('Bundle contains undeclared files')
        for field in ('entrypoint','model_path'):
            if field == 'entrypoint' and manifest.get(field) is None:
                continue
            name = safe_path(manifest.get(field,'')).removeprefix('payload/')
            if 'payload/'+name not in declared: raise ValueError('Missing '+field)
            manifest[field] = name
        destination.mkdir(parents=True, exist_ok=False)
        for archive, relative in declared.items():
            output = destination / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            with z.open(archive) as src, output.open('xb') as dst: shutil.copyfileobj(src,dst)
        (destination/'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def command(argv, cancelled=lambda:False, timeout=1800):
    import time
    with __import__('tempfile').TemporaryFile() as output:
        p = subprocess.Popen([str(x) for x in argv], stdout=output, stderr=subprocess.STDOUT)
        started = time.monotonic()
        while p.poll() is None:
            if cancelled() or time.monotonic()-started > timeout:
                p.terminate()
                try: p.wait(timeout=3)
                except subprocess.TimeoutExpired: p.kill(); p.wait()
                raise RuntimeError('Image operation cancelled or timed out')
            time.sleep(.1)
        output.seek(0); text = output.read(2*1024**2).decode(errors='replace')
    return p.returncode, text


def inject_partition(partition, payload, manifest, service=False, cancelled=lambda:False):
    """Use e2fsprogs against a regular partition file, never against devices."""
    if service and not manifest.get('entrypoint'): raise ValueError('Service installation requires an entrypoint')
    partition, payload = regular_file(partition), Path(payload).resolve()
    for tool in ('debugfs','e2fsck'):
        if not shutil.which(tool): raise RuntimeError('Missing Linux image backend: '+tool)
    code, output = command(['e2fsck','-f','-n',partition], cancelled)
    if code: raise RuntimeError('Base root filesystem is not clean: '+output[-3000:])
    def debug(request, write=False):
        code, text = command(['debugfs', *(['-w'] if write else []), '-R', request, partition], cancelled)
        if code or any(s in text.lower() for s in ('file not found','while opening','while writing','could not allocate','no space left','ext2fs_')):
            raise RuntimeError('debugfs operation failed: '+text[-3000:])
        return text
    debug('stat /etc/os-release')
    release = '/opt/pi-trainer/releases/'+manifest['release_id']
    def ensure_directory(path):
        code, output = command(['debugfs','-R','stat '+path,partition], cancelled)
        if 'Inode:' in output:
            if 'Type: directory' not in output: raise RuntimeError('Destination parent is not a directory: '+path)
        else: debug('mkdir '+path, True)
    def parents(path):
        current = ''
        for part in PurePosixPath(path).parts[1:]:
            current += '/'+part; ensure_directory(current)
    code, output = command(['debugfs','-R','stat '+release,partition],cancelled)
    if 'Inode:' in output: raise ValueError('Release already exists in image')
    parents(release)
    for source in sorted(payload.rglob('*')):
        if source.is_dir(): continue
        relative = safe_path(source.relative_to(payload).as_posix())
        target = release+'/'+relative
        parents(str(PurePosixPath(target).parent))
        debug('write "'+str(source)+'" '+target, True)
        if relative == manifest['entrypoint'] or relative.endswith('.sh'):
            debug('set_inode_field '+target+' mode 0100755', True)
        # Read each injected file back and compare, since debugfs can return 0 on failure.
        verification = payload.parent/'verify.bin'
        verification.unlink(missing_ok=True)
        debug('dump '+target+' "'+str(verification)+'"')
        if not verification.exists() or digest(verification) != digest(source):
            raise RuntimeError('Injected file verification failed: '+relative)
        verification.unlink()
    if service:
        unit = payload.parent/'pi-trainer.service'
        unit.write_text('[Unit]\nDescription=Pi Trainer model release\nAfter=network-online.target\n\n[Service]\nType=simple\nWorkingDirectory='+release+'\nExecStart='+release+'/'+manifest['entrypoint']+'\nRestart=on-failure\nUser=pi-trainer\nNoNewPrivileges=true\n\n[Install]\nWantedBy=multi-user.target\n')
        parents('/etc/systemd/system')
        code, output = command(['debugfs','-R','stat /etc/systemd/system/pi-trainer.service',partition],cancelled)
        if 'Inode:' in output: raise ValueError('Image already contains pi-trainer.service')
        debug('write "'+str(unit)+'" /etc/systemd/system/pi-trainer.service',True)
        # Installed disabled: user/runtime provisioning is intentionally explicit.
    code, output = command(['e2fsck','-f','-n',partition],cancelled)
    if code: raise RuntimeError('Modified filesystem failed consistency check: '+output[-3000:])


def customise_image(spec:dict, workdir:Path, emit=lambda message:None, cancelled=lambda:False)->dict:
    base = regular_file(spec['base_image'])
    if not (base.name.endswith('.img') or base.name.endswith('.img.xz')):
        raise ValueError('Base image must be .img or .img.xz')
    job = Path(workdir).resolve() / ('image-'+uuid.uuid4().hex)
    # debugfs request quoting: refuse quotes/newlines in host work directory.
    if ',' in str(job) or any(c in str(job) for c in ('"','\n','\r')): raise ValueError('Unsupported workspace path')
    job.mkdir(parents=True,exist_ok=False)
    payload = job/'payload'
    manifest = unpack_bundle(spec['bundle_path'],payload)
    output = job/'customised.img'
    maximum = int(spec.get('max_image_bytes',128*1024**3))
    if maximum <= 0 or maximum > 2*1024**4: raise ValueError('Invalid image size bound')
    emit('Copying base image; original is preserved')
    try:
        opener = lzma.open if base.name.endswith('.xz') else open
        with opener(base,'rb') as src, output.open('xb') as dst:
            total=0
            while True:
                if cancelled(): raise RuntimeError('Image operation cancelled')
                block=src.read(4*1024**2)
                if not block: break
                total += len(block)
                if total > maximum: raise ValueError('Expanded image exceeds configured limit')
                dst.write(block)
        part=select_partition(inspect_partitions(output),spec.get('partition'))
        partition=job/'root.ext4'
        with output.open('rb') as src, partition.open('xb') as dst:
            src.seek(part['offset']); remaining=part['size']
            while remaining:
                if cancelled(): raise RuntimeError('Image operation cancelled')
                block=src.read(min(4*1024**2,remaining))
                if not block: raise ValueError('Truncated partition')
                dst.write(block); remaining-=len(block)
        emit('Injecting and verifying release in isolated root filesystem')
        service=bool(spec.get('install_service',False))
        if sys.platform.startswith('linux') and shutil.which('debugfs') and shutil.which('e2fsck'):
            inject_partition(partition,payload,manifest,service,cancelled)
            backend='linux-e2fsprogs'
        else:
            if not shutil.which('docker'): raise RuntimeError('Install Docker and build pi-trainer-image-worker:local; see docs/images.md')
            worker=spec.get('worker_image','pi-trainer-image-worker:local')
            if not isinstance(worker,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._/:@-]+',worker): raise ValueError('Invalid Docker worker image')
            code, detail=command(['docker','image','inspect',worker],cancelled,30)
            if code: raise RuntimeError('Docker image worker unavailable; build docker/image-worker.Dockerfile first: '+detail[-1500:])
            package=Path(__file__).resolve().parent
            container='pi-trainer-image-'+uuid.uuid4().hex
            argv=['docker','run','--rm','--name',container,'--network','none','--cap-drop','ALL','--security-opt','no-new-privileges','--mount','type=bind,src='+str(job)+',dst=/job','--mount','type=bind,src='+str(package)+',dst=/app/pi_trainer,readonly',worker,'python','/app/pi_trainer/image_worker.py','/job',*(['--service'] if service else [])]
            try: code, detail=command(argv,cancelled)
            finally:
                # A cancelled Docker client does not guarantee the container stopped.
                command(['docker','rm','-f',container],timeout=30)
            if code: raise RuntimeError('Docker image worker failed: '+detail[-3000:])
            backend='docker-e2fsprogs'
        if partition.stat().st_size != part['size']: raise RuntimeError('Partition size changed unexpectedly')
        with output.open('r+b') as dst, partition.open('rb') as src:
            dst.seek(part['offset'])
            while block:=src.read(4*1024**2):
                if cancelled(): raise RuntimeError('Image operation cancelled')
                dst.write(block)
        partition.unlink()
        result=dict(status='completed',image_path=str(output),sha256=digest(output),bytes=output.stat().st_size,base_image=str(base),release_id=manifest['release_id'],partition=part['number'],backend=backend,service_installed=service,service_enabled=False,hardware_tested=False)
        (job/'result.json').write_text(json.dumps(result,indent=2))
        emit('Image copy completed and checksummed')
        return result
    except Exception:
        output.unlink(missing_ok=True)
        raise
