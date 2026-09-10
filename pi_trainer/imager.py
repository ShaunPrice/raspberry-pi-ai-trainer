"""Prepare an offline Imager catalogue; never select or erase a disk."""
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

def executable():
    candidates=[shutil.which('rpi-imager')]
    if platform.system()=='Darwin':candidates.append('/Applications/Raspberry Pi Imager.app/Contents/MacOS/rpi-imager')
    elif platform.system()=='Windows':
        candidates.extend([str(Path(os.environ.get('ProgramFiles','C:/Program Files'))/'Raspberry Pi Imager/rpi-imager.exe'),str(Path(os.environ.get('ProgramFiles(x86)','C:/Program Files (x86)'))/'Raspberry Pi Imager/rpi-imager.exe')])
    return next((str(p) for p in candidates if p and Path(p).is_file()),None)

def prepare(store,image_path):
    from .images import inspect_partitions
    image=Path(image_path).expanduser().resolve(strict=True)
    if not image.is_file() or image.suffix!='.img':raise ValueError('Select the customised raw .img output')
    inspect_partitions(image)
    with image.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    repo=store.exports/(digest[:16]+'-imager.json')
    catalogue={'imager':{'latest_version':'2.0.0','url':'https://www.raspberrypi.com/software/','devices':[{'name':'Raspberry Pi 5','tags':['pi5'],'default':True,'matching_type':'inclusive'}]},
               'os_list':[{'name':'Pi Trainer · '+image.name,'description':'Customised existing image. Choose and verify removable media in Imager.','url':image.as_uri(),'extract_size':image.stat().st_size,'extract_sha256':digest,'image_download_size':image.stat().st_size,'image_download_sha256':digest,'release_date':time.strftime('%Y-%m-%d'),'init_format':'none','devices':['pi5']}]}
    repo.write_text(json.dumps(catalogue,indent=2))
    binary=executable()
    return {'image_path':str(image),'sha256':digest,'catalogue_path':str(repo),'command':[binary,'--repo',str(repo)] if binary else None,'imager_installed':bool(binary),'disk_selected':False,'media_written':False}

def open_imager(store,image_path):
    result=prepare(store,image_path)
    if not result['command']:raise RuntimeError('Install Raspberry Pi Imager, then open the generated catalogue with --repo')
    log=store.root/'logs';log.mkdir(exist_ok=True)
    with (log/'imager.log').open('ab') as stream:subprocess.Popen(result['command'],stdin=subprocess.DEVNULL,stdout=stream,stderr=stream)
    return {**result,'imager_launched':True,'note':'Choose storage and confirm writing in Raspberry Pi Imager; this operation does not erase media.'}
