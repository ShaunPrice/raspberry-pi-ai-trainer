import json
import tempfile
import unittest
from pathlib import Path
from pi_trainer.llm_tools import score_outputs,prepare_recipe,estimate

class ReuseTests(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def test_missing_outputs_fail_and_boolean_is_not_number(self):
        tasks=self.root/'tasks.jsonl';outputs=self.root/'outputs.json'
        tasks.write_text('\n'.join(json.dumps(x) for x in [
            {'id':'a','prompt':'JSON one','expected':1,'evaluator':'json'},
            {'id':'b','prompt':'Say yes','expected':'yes'}]))
        outputs.write_text(json.dumps({'a':'true'}))
        result=score_outputs(tasks,outputs);self.assertEqual(result['quality'],0);self.assertTrue(result['tasks'][1]['missing'])
        outputs.write_text(json.dumps({'a':'1','b':'yes'}));self.assertEqual(score_outputs(tasks,outputs)['quality'],1)
    def test_recipe_reuses_engine_schema_and_never_claims_compilation(self):
        data=self.root/'alpaca.jsonl';data.write_text('\n'.join(json.dumps({'instruction':f'Question {i}','output':f'Answer {i}'}) for i in range(20)))
        output=self.root/'recipe.json';result=prepare_recipe('soup-mlx','local/model',str(data),str(output))
        self.assertFalse(result['training_run']);self.assertEqual(result['config']['backend'],'mlx')
        self.assertIn('vendor recipe',result['hailo_compilation'])
        with self.assertRaises(FileExistsError):prepare_recipe('soup-mlx','local/model',str(data),str(output))
    def test_memory_estimate_is_not_hailo_fit(self):
        r=estimate(parameters_b=1.5,weight_bits=4,layers=28,kv_heads=2,head_dim=128,context=2048)
        self.assertGreater(r['estimated_total_gib'],0)
        self.assertIn('not a fit guarantee',r['evidence'])
        with self.assertRaises(ValueError):estimate(parameters_b=float('nan'),weight_bits=4,layers=28,kv_heads=2,head_dim=128,context=2048)
