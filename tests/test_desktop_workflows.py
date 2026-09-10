import unittest
from pi_trainer.desktop_workflows import KINDS, template, parse_spec

class WorkflowUiTests(unittest.TestCase):
    def test_templates_cover_workflows_without_execution(self):
        for kind in KINDS:
            self.assertIsInstance(template(kind),dict)
        self.assertFalse(template('image')['install_service'])
        self.assertFalse(template('train','llm')['allow_download'])
        self.assertEqual(template('train','llm')['engine'],'transformers-peft')
        self.assertIn('recipe_executable',template('compile','llm'))
    def test_spec_requires_json_object_and_finite_numbers(self):
        for text in ('[]','null','{"epochs":NaN}','{"epochs":Infinity}'):
            with self.assertRaises(ValueError):parse_spec(text)
        self.assertEqual(parse_spec('{"epochs":2}'),{'epochs':2})
    def test_templates_are_independent(self):
        value=template('optimise');value['search']['epochs'].append(999)
        self.assertEqual(template('optimise')['search']['epochs'],[1])

if __name__=='__main__':unittest.main()
