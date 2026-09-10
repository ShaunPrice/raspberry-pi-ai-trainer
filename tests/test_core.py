import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from pi_trainer.core import Store

class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve();self.store=Store(self.root/'state')
    def tearDown(self):self.temp.cleanup()
    def corpus(self):
        path=self.root/'data.jsonl';path.write_text('\n'.join(json.dumps({'text':f'Example {i}'}) for i in range(100)))
        return str(path)
    def test_target_gating_and_persistence(self):
        for target in ['hailo8','hailo8l']:
            with self.assertRaises(ValueError):self.store.create_project('x',target,'llm')
        p=self.store.create_project('LLM','hailo10h','llm')
        self.assertEqual(Store(self.store.root).list_projects()[0],p)
    def test_record_splits_snapshot_and_bundle(self):
        p=self.store.create_project('LLM','hailo10h','llm');source=self.corpus()
        d=self.store.import_dataset(p['id'],source)
        self.assertEqual(d['count'],100);self.assertTrue(all(d['split_counts'].values()))
        sets={s:{r['sha256'] for r in d['items'] if r['split']==s} for s in ['train','validation','test']}
        self.assertFalse(sets['train']&sets['test']);self.assertFalse(sets['validation']&sets['test'])
        self.assertTrue(all(r['split']=='train' for r in d['items'] if r['calibration']))
        again=self.store.import_dataset(p['id'],source);self.assertEqual(d['sha256'],again['sha256'])
        Path(source).write_text('changed')
        blob=self.store.blobs/d['files'][0]['sha256'];self.assertNotEqual(blob.read_text(),'changed')
        b=self.store.bundle(p['id']);self.assertFalse(b['compiled_model']);self.assertFalse(b['bootable_image'])
        with zipfile.ZipFile(b['path']) as z:
            checks=json.loads(z.read('checksums.json'))
            for name,digest in checks.items():self.assertEqual(hashlib.sha256(z.read(name)).hexdigest(),digest)
    def test_duplicate_records_never_cross_splits(self):
        p=self.store.create_project('LLM','hailo10h','llm');source=Path(self.corpus())
        with source.open('a') as f:f.write('\n'+json.dumps({'text':'Example 0'}))
        d=self.store.import_dataset(p['id'],str(source));self.assertEqual(d['duplicates_removed'],1);self.assertEqual(d['count'],100)
    def test_corruption_blocks_bundle(self):
        p=self.store.create_project('LLM','hailo10h','llm');d=self.store.import_dataset(p['id'],self.corpus())
        (self.store.blobs/d['files'][0]['sha256']).write_bytes(b'bad')
        with self.assertRaises(ValueError):self.store.bundle(p['id'])
    def test_no_fake_execution(self):
        p=self.store.create_project('Vision','hailo8l','vision');plan=self.store.plan(p['id'])
        self.assertTrue(all(s['status']=='blocked' for s in plan['stages']))
        self.assertEqual(self.store.list_jobs(),[])
    def test_invalid_jsonl_leaves_no_dataset(self):
        p=self.store.create_project('LLM','hailo10h','llm');source=self.root/'bad.jsonl';source.write_text('{"messages": [{"role":"bogus","content":"x"}]}')
        with self.assertRaises(ValueError):self.store.import_dataset(p['id'],str(source))
        self.assertEqual(self.store.datasets(p['id']),[])
    def test_self_import_and_symlinks_rejected(self):
        p=self.store.create_project('LLM','hailo10h','llm')
        with self.assertRaises(ValueError):self.store.import_dataset(p['id'],str(self.store.root))
        link=self.root/'link.jsonl'
        try:link.symlink_to(self.corpus())
        except OSError:self.skipTest('Symlink creation unavailable')
        with self.assertRaises(ValueError):self.store.import_dataset(p['id'],str(link))
    def test_bad_image_signature_rejected(self):
        p=self.store.create_project('Vision','hailo8','vision');file=self.root/'bad.png';file.write_text('not an image')
        with self.assertRaises(ValueError):self.store.import_dataset(p['id'],str(file))

    def test_host_recipe_excludes_held_out_records(self):
        p=self.store.create_project('LLM','hailo10h','llm');source=self.root/'alpaca.jsonl'
        source.write_text('\n'.join(json.dumps({'instruction':f'Question {i}','output':f'Answer {i}'}) for i in range(100)))
        result=self.store.llm_recipe(p['id'],'soup-mlx','example/model',str(source))
        dataset=self.store.datasets(p['id'])[-1]
        training=Path(result['config']['data']['train']).read_text().splitlines()
        held={item['sha256'] for item in dataset['items'] if item['split']!='train'}
        from pi_trainer.core import encode
        self.assertTrue(training)
        self.assertFalse({hashlib.sha256(encode(json.loads(line)).encode()).hexdigest() for line in training}&held)
        self.assertEqual(result['training_records'],len(training))
