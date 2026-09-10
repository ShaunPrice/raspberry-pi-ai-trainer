"""Reusable LLM-Optimise preparation/scoring tools; no model runs or downloads."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from .vendor.llm_optimise.quality import load_tasks
from .vendor.llm_optimise.training import training_recipe, save_recipe
from .vendor.llm_optimise.planner import estimate_memory


def score_outputs(tasks_path, outputs_path):
    paths=[Path(tasks_path),Path(outputs_path)]
    if any(not p.is_file() or p.stat().st_size>16*1024*1024 for p in paths):
        raise ValueError('Task and output files must be regular files no larger than 16 MiB')
    tasks=load_tasks(paths[0])
    outputs=json.loads(paths[1].read_text(encoding='utf-8'))
    if not isinstance(outputs,dict) or any(not isinstance(v,str) for v in outputs.values()):
        raise ValueError('Outputs must be a JSON object mapping task IDs to response strings')
    known={t.id for t in tasks}
    if set(outputs)-known:raise ValueError('Outputs contain unknown task IDs')
    if any(t.json_schema is not None for t in tasks):
        raise ValueError('JSON-schema gates require the full LLM-Optimise workbench; use exact/json/json_subset/numeric evaluators here')
    scores=[{'id':t.id,'score':t.score(outputs[t.id]) if t.id in outputs else 0.0,
             'missing':t.id not in outputs} for t in tasks]
    return {'schema':'pi-trainer/quality/v1','quality':sum(s['score'] for s in scores)/len(scores),
            'tasks':scores,'evidence':'scoring supplied outputs only; inference was not run',
            'input_sha256':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            'provider':'LLM-Optimise MIT quality utilities'}


def prepare_recipe(engine, model, data, output, model_output='adapters', max_length=512, rank=8):
    source=Path(data).resolve()
    if not source.is_file() or source.stat().st_size>16*1024*1024:
        raise ValueError('Use a reviewed Alpaca-format JSONL file no larger than 16 MiB')
    lines=[json.loads(line) for line in source.read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(lines)<2 or any(not isinstance(r,dict) or not isinstance(r.get('instruction'),str) or not r['instruction'].strip() or not isinstance(r.get('output'),str) or not r['output'].strip() or ('input' in r and not isinstance(r['input'],str)) for r in lines):
        raise ValueError('Recipe requires at least two Alpaca records with nonempty instruction and output strings')
    recipe=training_recipe(engine,model,str(source),str(Path(model_output).resolve()),max_length=max_length,rank=rank)
    recipe['notes'] += ['Hailo-10H compilation and on-device adapter compatibility are not established by this host training recipe.',
                        'This recipe uses the upstream validation split; reserve a separate final test set and review group leakage before training.']
    result=save_recipe(recipe,output)
    return {**result,'provider':'LLM-Optimise MIT training recipe utilities','training_run':False,
            'hailo_compilation':'requires qualified vendor recipe','data_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}


def estimate(**parameters):
    result=estimate_memory(**parameters)
    result['assumptions'].append('Not a Hailo compiler memory-layout prediction, training memory estimate or 10H fit guarantee.')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);subs=parser.add_subparsers(dest='action',required=True)
    score=subs.add_parser('score');score.add_argument('tasks');score.add_argument('outputs')
    recipe=subs.add_parser('recipe')
    recipe.add_argument('--engine',choices=['soup-mlx','soup-qlora','soup-stream'],required=True)
    for name in ['model','data','output']:recipe.add_argument('--'+name,required=True)
    recipe.add_argument('--model-output',default='adapters');recipe.add_argument('--max-length',type=int,default=512);recipe.add_argument('--rank',type=int,default=8)
    est=subs.add_parser('estimate');est.add_argument('parameters',help='JSON parameter file for the dense-transformer memory estimator')
    args=vars(parser.parse_args());action=args.pop('action')
    try:
        if action=='score':result=score_outputs(args['tasks'],args['outputs'])
        elif action=='recipe':result=prepare_recipe(**args)
        else:result=estimate(**json.loads(Path(args['parameters']).read_text()))
        print(json.dumps(result,indent=2,allow_nan=False))
    except (ValueError,TypeError,OSError) as exc:parser.exit(1,json.dumps({'error':str(exc)})+'\n')


if __name__=='__main__':main()
