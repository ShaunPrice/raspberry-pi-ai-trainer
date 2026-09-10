import hashlib
import json
import lzma
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import zlib

from pi_trainer.images import (customise_image, inspect_partitions, select_partition,
    unpack_bundle, inject_partition, LINUX_GUID)


def bundle(path,entrypoint='run.sh'):
    content={'model.hef':b'compiled model fixture','run.sh':b'#!/bin/sh\nexit 0\n'}
    manifest=dict(schema='pi-trainer/deployment/v1',release_id='abc123',target='hailo8',runtime_version='4.22',entrypoint=entrypoint,model_path='model.hef',files=[dict(path=k,sha256=hashlib.sha256(v).hexdigest(),bytes=len(v)) for k,v in content.items()])
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('manifest.json',json.dumps(manifest))
        for k,v in content.items():z.writestr('payload/'+k,v)
    return manifest


def mbr_image(path,partition=None):
    payload=partition.read_bytes() if partition else bytes(8192)
    data=bytearray(2048+len(payload));data[510:512]=b'\x55\xaa'
    data[450]=0x83;struct.pack_into('<II',data,454,4,len(payload)//512)
    data[2048:]=payload
    if not partition:data[2048+1080:2048+1082]=b'\x53\xef'
    path.write_bytes(data)


class ImageTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
    def test_mbr_bounds_and_ambiguity(self):
        p=self.root/'base.img';mbr_image(p)
        parts=inspect_partitions(p)
        self.assertEqual(select_partition(parts)['offset'],2048)
        with self.assertRaises(ValueError):select_partition(parts+parts)
        bad=bytearray(p.read_bytes());struct.pack_into('<I',bad,458,100000);p.write_bytes(bad)
        with self.assertRaisesRegex(ValueError,'boundary'):inspect_partitions(p)
    def test_gpt_crc(self):
        image=bytearray(512*100)
        image[510:512]=b'\x55\xaa';image[450]=0xee
        table=bytearray(128);table[:16]=LINUX_GUID
        struct.pack_into('<QQ',table,32,10,89)
        header=bytearray(512);header[:8]=b'EFI PART'
        struct.pack_into('<II',header,8,0x10000,92)
        struct.pack_into('<QQQQ',header,24,1,99,10,89)
        struct.pack_into('<QIII',header,72,2,1,128,zlib.crc32(table))
        struct.pack_into('<I',header,16,zlib.crc32(header[:92]))
        image[512:1024]=header;image[1024:1152]=table
        image[10*512+1080:10*512+1082]=b'\x53\xef'
        p=self.root/'gpt.img';p.write_bytes(image)
        self.assertEqual(select_partition(inspect_partitions(p))['number'],1)
        image[1040]^=1;p.write_bytes(image)
        with self.assertRaisesRegex(ValueError,'CRC'):inspect_partitions(p)
    def test_bundle_validation(self):
        z=self.root/'bundle.zip';bundle(z)
        manifest=unpack_bundle(z,self.root/'payload')
        self.assertEqual(manifest['release_id'],'abc123')
        with zipfile.ZipFile(z,'a') as archive:archive.writestr('../escape',b'bad')
        with self.assertRaises(ValueError):unpack_bundle(z,self.root/'bad')
        self.assertFalse((self.root/'bad').exists())
    def test_hash_and_symlink_rejected(self):
        for kind in ('hash','symlink'):
            z=self.root/(kind+'.zip');manifest=bundle(z)
            with zipfile.ZipFile(z) as src:entries={n:src.read(n) for n in src.namelist()}
            with zipfile.ZipFile(z,'w') as out:
                for name,data in entries.items():
                    if kind=='hash' and name=='payload/model.hef':data=b'x'*len(data)
                    if kind=='symlink' and name=='payload/model.hef':
                        info=zipfile.ZipInfo(name);info.external_attr=0o120777<<16;out.writestr(info,data)
                    else:out.writestr(name,data)
            with self.assertRaises(ValueError):unpack_bundle(z,self.root/kind)
    def test_original_preserved_when_backend_fails(self):
        base=self.root/'base.img';mbr_image(base);original=base.read_bytes()
        z=self.root/'bundle.zip';bundle(z)
        with patch('pi_trainer.images.shutil.which',return_value=None):
            with self.assertRaises(RuntimeError):customise_image(dict(base_image=str(base),bundle_path=str(z)),self.root)
        self.assertEqual(base.read_bytes(),original)
        self.assertEqual(list(self.root.glob('image-*/customised.img')),[])
    def test_xz_copy_and_partition_preservation(self):
        base=self.root/'base.img';mbr_image(base);original=base.read_bytes()
        compressed=self.root/'base.img.xz';compressed.write_bytes(lzma.compress(original))
        z=self.root/'bundle.zip';bundle(z,entrypoint=None)
        def inject(path,*args):
            with open(path,'r+b') as f:f.seek(4096);f.write(b'MODEL')
        with patch('pi_trainer.images.sys.platform','linux'),patch('pi_trainer.images.shutil.which',return_value='/bin/tool'),patch('pi_trainer.images.inject_partition',side_effect=inject):
            result=customise_image(dict(base_image=str(compressed),bundle_path=str(z)),self.root)
        output=Path(result['image_path']).read_bytes()
        self.assertEqual(output[:2048],original[:2048]);self.assertEqual(len(output),len(original))
        self.assertEqual(output[2048+4096:2048+4101],b'MODEL')
        self.assertEqual(base.read_bytes(),original)
        self.assertEqual(result['sha256'],hashlib.sha256(output).hexdigest())
    @unittest.skipUnless(os.environ.get('PI_TRAINER_TEST_DOCKER') == '1', 'opt-in Docker end-to-end test')
    def test_host_docker_backend(self):
        partition=self.root/'root.ext4'
        with partition.open('wb') as f:f.truncate(32*1024**2)
        (self.root/'os-release').write_text('ID=raspbian\n')
        subprocess.run(['docker','run','--rm','--network','none','--cap-drop','ALL',
            '--mount','type=bind,src='+str(self.root.resolve())+',dst=/job',
            'pi-trainer-image-worker:local','sh','-c',
            'mkfs.ext4 -q -F /job/root.ext4 && debugfs -w -R "mkdir /etc" /job/root.ext4 && debugfs -w -R "write /job/os-release /etc/os-release" /job/root.ext4'], check=True,capture_output=True)
        base=self.root/'base.img';mbr_image(base,partition);original=hashlib.sha256(base.read_bytes()).hexdigest()
        z=self.root/'bundle.zip';bundle(z)
        # Force the exact Docker branch used on Mac/Windows even on Linux test hosts.
        with patch('pi_trainer.images.sys.platform','darwin'):
            result=customise_image(dict(base_image=str(base),bundle_path=str(z),install_service=True),self.root)
        self.assertEqual(result['backend'],'docker-e2fsprogs')
        self.assertEqual(hashlib.sha256(base.read_bytes()).hexdigest(),original)
        self.assertNotEqual(result['sha256'],original)

    @unittest.skipUnless(shutil.which('debugfs') and shutil.which('mkfs.ext4') and shutil.which('e2fsck'),'requires Linux e2fsprogs')
    def test_real_ext4_injection(self):
        partition=self.root/'root.ext4'
        with partition.open('wb') as f:f.truncate(32*1024**2)
        subprocess.run(['mkfs.ext4','-q','-F',str(partition)],check=True)
        osrelease=self.root/'os-release';osrelease.write_text('ID=raspbian\n')
        for request in ('mkdir /etc','write '+str(osrelease)+' /etc/os-release'):
            subprocess.run(['debugfs','-w','-R',request,str(partition)],check=True,capture_output=True)
        base=self.root/'base.img';mbr_image(base,partition);before=hashlib.sha256(base.read_bytes()).hexdigest()
        z=self.root/'bundle.zip';bundle(z)
        result=customise_image(dict(base_image=str(base),bundle_path=str(z),install_service=True),self.root)
        self.assertEqual(hashlib.sha256(base.read_bytes()).hexdigest(),before)
        self.assertNotEqual(result['sha256'],before)
        self.assertFalse(result['hardware_tested'])

if __name__=='__main__':unittest.main()
