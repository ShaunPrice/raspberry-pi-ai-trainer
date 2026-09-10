"""Cross-service integrity and durable execution tests without SDKs or hardware."""
import base64
import hashlib
import json
from pathlib import Path
import stat
import struct
import tempfile
import time
import unittest
from unittest.mock import patch
import zipfile

from pi_trainer.core import Store
from pi_trainer import jobs
from pi_trainer.data import materialise
from pi_trainer.operations import dispatch
from pi_trainer.remote import safe_extract


class WorkflowIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name).resolve()
        self.store=Store(self.root/'state')
        self.project=self.store.create_project('Integration','hailo10h','llm')['id']

    def dataset(self):
        source=self.root/'corpus.jsonl'
        source.write_text('\n'.join(json.dumps({'instruction':f'Question {i}', 'output':f'Answer {i}'}) for i in range(100)))
        return self.store.import_dataset(self.project,str(source))

    def settle(self,jid,seconds=20):
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:
            job=self.store.job_get(jid)
            if job['status'] in jobs.TERMINAL:
                # Let detached worker relinquish its SQLite lease before temp cleanup.
                while time.monotonic()<deadline:
                    with self.store.connection() as db:active=db.execute('SELECT COUNT(*) FROM worker_lock').fetchone()[0]
                    if not active:return job
                    time.sleep(.05)
                self.fail('Worker did not release its lease')
            time.sleep(.05)
        self.store.job_cancel(jid)
        self.fail('Durable job did not settle within test deadline: '+json.dumps(self.store.job_get(jid)))

    def test_real_detached_export_survives_store_reopen(self):
        dataset=self.dataset()
        job=dispatch(self.store,'run_job',{'project_id':self.project,'kind':'export_dataset','spec':{'dataset_id':dataset['id']}})
        self.store=Store(self.store.root)
        result=self.settle(job['id'])
        self.assertEqual(result['status'],'succeeded',result.get('error'))
        archive=Path(result['result']['archive_path'])
        self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(),result['result']['sha256'])
        with zipfile.ZipFile(archive) as z:
            snapshot=json.loads(z.read('snapshot.json'))
            self.assertEqual(snapshot['dataset_id'],dataset['id'])
            records=[json.loads(line) for split in ('train','validation','test') for line in z.read(split+'.jsonl').decode().splitlines()]
            self.assertEqual(len(records),100)
        self.assertEqual(self.store.job_logs(job['id'])['job']['status'],'succeeded')

    def test_queued_job_pins_original_dataset(self):
        original=self.dataset()
        with patch('pi_trainer.jobs.spawn_worker'):
            job=self.store.run_job(self.project,'export_dataset',{})
        replacement=self.dataset()
        self.assertNotEqual(original['id'],replacement['id'])
        self.assertEqual(job['spec']['dataset_id'],original['id'])
        jobs.worker_loop(self.store)
        self.assertEqual(self.store.job_get(job['id'])['result']['dataset_id'],original['id'])

    def test_queued_cancellation_never_executes(self):
        self.dataset()
        with patch('pi_trainer.jobs.spawn_worker'):
            job=self.store.run_job(self.project,'export_dataset',{})
        cancelled=self.store.job_cancel(job['id'])
        self.assertEqual(cancelled['status'],'cancelled')
        jobs.worker_loop(self.store)
        self.assertFalse((self.store.root/'runs'/job['id']).exists())
        self.assertEqual(self.store.job_get(job['id'])['status'],'cancelled')

    def test_stale_running_job_is_interrupted_not_replayed(self):
        with patch('pi_trainer.jobs.spawn_worker'):
            job=self.store.run_job(self.project,'deploy',{'device_id':'not-contacted'})
        jobs.update(self.store,job['id'],status='running')
        jobs.worker_loop(self.store)
        recovered=self.store.job_get(job['id'])
        self.assertEqual(recovered['status'],'interrupted')
        self.assertIn('Inspect logs',recovered['error'])
        self.assertFalse((self.store.root/'runs'/job['id']).exists())

    def test_browser_upload_checks_paths_project_and_hash(self):
        upload=self.store.upload_create(self.project)['id']
        content=b'{"instruction":"Question", "output":"Answer"}\n'
        encoded=base64.b64encode(content).decode()
        for name in ('../outside.jsonl','/outside.jsonl','folder/../../outside.jsonl','C:/outside.jsonl','folder\\outside.jsonl'):
            with self.subTest(name=name),self.assertRaises(ValueError):
                self.store.upload_file(self.project,upload,name,encoded)
        other=self.store.create_project('Other','hailo10h','llm')['id']
        with self.assertRaises(ValueError):self.store.upload_file(other,upload,'data.jsonl',encoded)
        with self.assertRaises(ValueError):self.store.upload_file(self.project,upload,'data.jsonl','!!!')
        self.store.upload_file(self.project,upload,'reviewed/data.jsonl',encoded)
        with self.assertRaises(FileExistsError):self.store.upload_file(self.project,upload,'reviewed/data.jsonl',encoded)
        dataset=self.store.upload_finish(self.project,upload)
        self.assertEqual(dataset['files'][0]['sha256'],hashlib.sha256(content).hexdigest())
        self.assertEqual(self.store.upload_finish(self.project,upload)['id'],dataset['id'])
        with self.assertRaises(ValueError):self.store.upload_file(self.project,upload,'later.jsonl',encoded)
        self.assertFalse((self.root/'outside.jsonl').exists())

    def test_corrupt_snapshot_fails_real_job_without_success_artifact(self):
        dataset=self.dataset();(self.store.blobs/dataset['files'][0]['sha256']).write_bytes(b'corrupt')
        job=self.store.run_job(self.project,'export_dataset',{'dataset_id':dataset['id']})
        result=self.settle(job['id'])
        self.assertEqual(result['status'],'failed')
        self.assertIn('integrity',result['error'])
        self.assertIsNone(result['result'])
        self.assertFalse((Path(result['workdir'])/'dataset.zip').exists())

    def test_group_annotation_versions_do_not_mutate_snapshot(self):
        dataset=self.dataset();before=json.dumps(dataset,sort_keys=True)
        annotations={item['sha256']:{'group':'same-source'} for item in dataset['items'][:8]}
        path=self.root/'annotations.json';path.write_text(json.dumps(annotations))
        updated=self.store.annotate(self.project,dataset['id'],str(path))
        self.assertNotEqual(updated['id'],dataset['id'])
        self.assertEqual(updated['parent_dataset_id'],dataset['id'])
        grouped=[i for i in updated['items'] if i.get('group')=='same-source']
        self.assertEqual(len({i['split'] for i in grouped}),1)
        original=next(d for d in self.store.datasets(self.project) if d['id']==dataset['id'])
        self.assertEqual(json.dumps(original,sort_keys=True),before)
        self.assertTrue(all(i['calibration']==(i['split']=='train') for i in updated['items']))

    def test_script_save_never_executes_and_retains_previous_hash(self):
        sentinel=self.root/'must-not-exist'
        source='from pathlib import Path\nPath('+repr(str(sentinel))+').touch()\n'
        first=dispatch(self.store,'write_script',{'project_id':self.project,'name':'run.py','content':source})
        self.assertFalse(first['executed']);self.assertFalse(sentinel.exists())
        second=self.store.write_script(self.project,'run.py','# revision\n')
        self.assertEqual(second['previous_sha256'],first['sha256'])
        self.assertNotEqual(second['sha256'],first['sha256'])
        self.assertEqual(self.store.project_files(self.project)['files'][0]['name'],'run.py')
        for name in ('../run.py','/tmp/run.py','run.exe','x/y.py'):
            with self.assertRaises(ValueError):self.store.write_script(self.project,name,'bad')

    def test_imager_catalogue_is_hash_bound_and_never_launches(self):
        image=self.root/'existing image.img';data=bytearray(12288)
        data[510:512]=b'\x55\xaa';data[450]=0x83
        struct.pack_into('<II',data,454,4,20);data[2048+1080:2048+1082]=b'\x53\xef';image.write_bytes(data)
        with patch('pi_trainer.imager.executable',return_value='/imaginary/imager'),patch('pi_trainer.imager.subprocess.Popen') as spawn:
            result=self.store.prepare_imager(str(image))
        spawn.assert_not_called()
        self.assertFalse(result['disk_selected']);self.assertFalse(result['media_written'])
        catalog=json.loads(Path(result['catalogue_path']).read_text())
        entry=catalog['os_list'][0]
        self.assertEqual(entry['url'],image.as_uri());self.assertEqual(entry['extract_sha256'],hashlib.sha256(data).hexdigest())
        self.assertEqual(result['command'],['/imaginary/imager','--repo',result['catalogue_path']])
        self.assertEqual(image.read_bytes(),bytes(data))

    def test_remote_archive_security_and_honest_registration(self):
        for name,mode in (('../escape',0),('/escape',0),('C:/escape',0),('link',stat.S_IFLNK|0o777)):
            archive=self.root/'bad.zip'
            with zipfile.ZipFile(archive,'w') as z:
                entry=zipfile.ZipInfo(name);entry.external_attr=mode<<16;z.writestr(entry,b'bad')
            with self.subTest(name=name),self.assertRaises(ValueError):safe_extract(archive,self.root/'extract')
        self.assertFalse((self.root/'escape').exists())
        worker=self.store.worker_enroll('Compiler','trusted-linux','/opt/sdk/bin/python')
        self.assertEqual(worker['status'],'registered_not_verified')
        for host in ('-oProxyCommand=bad','host;command','user@host'):
            with self.assertRaises(ValueError):self.store.worker_enroll('Bad',host,'python3')
        with self.assertRaises(ValueError):self.store.worker_enroll('Bad','host','python3;bad')

    def test_shared_operation_schema_rejects_unknown_or_extra_arguments(self):
        with self.assertRaises(ValueError):dispatch(self.store,'run_shell',{'command':'touch never'})
        with self.assertRaises(ValueError):dispatch(self.store,'workflow_jobs',{'extra':True})
        with self.assertRaises(ValueError):dispatch(self.store,'run_job',{'project_id':self.project,'kind':'train','spec':[]})
        with self.assertRaises(ValueError):self.store.run_job(self.project,'train',{'target':'hailo8'})
        self.assertEqual(self.store.workflow_jobs(),[])

if __name__=='__main__':unittest.main()
