"""Pi Trainer command line."""
from __future__ import annotations
import argparse
import json
import time
import sys
from pathlib import Path
from .core import Store

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-dir',default='.trainer')
    sub=p.add_subparsers(dest='action',required=True)
    for name in ['capabilities','projects','jobs','mcp','desktop']:sub.add_parser(name)
    web=sub.add_parser('web');web.add_argument('--port',type=int,default=8765)
    create=sub.add_parser('create');create.add_argument('--name',required=True);create.add_argument('--target',choices=['hailo8l','hailo8','hailo10h'],required=True);create.add_argument('--task',choices=['vision','llm'],required=True)
    imp=sub.add_parser('import');imp.add_argument('project_id');imp.add_argument('path')
    for name in ['plan','bundle','datasets']:sub.add_parser(name).add_argument('project_id')
    recipe=sub.add_parser('llm-recipe');recipe.add_argument('project_id')
    for name in ['engine','model','data']:recipe.add_argument('--'+name,required=True)
    score=sub.add_parser('score');score.add_argument('tasks_path');score.add_argument('outputs_path')
    estimate=sub.add_parser('estimate');estimate.add_argument('parameters_file')
    for name in ['workflow-jobs','doctor']:sub.add_parser(name)
    for name in ['job','logs','cancel']:sub.add_parser(name).add_argument('job_id')
    run=sub.add_parser('run');run.add_argument('project_id');run.add_argument('kind');run.add_argument('--spec',required=True);run.add_argument('--wait',action='store_true')
    invoke=sub.add_parser('invoke');invoke.add_argument('operation');invoke.add_argument('--arguments',required=True)
    args=vars(p.parse_args(argv));store=Store(Path(args.pop('data_dir')));action=args.pop('action')
    try:
        if action=='invoke':
            from .operations import dispatch
            result=dispatch(store,args['operation'],json.loads(Path(args['arguments']).read_text()))
            print(json.dumps(result,indent=2,allow_nan=False));return 0
        if action=='run':
            from .jobs import TERMINAL
            result=store.run_job(args['project_id'],args['kind'],json.loads(Path(args['spec']).read_text()))
            if args['wait']:
                previous=None
                while result['status'] not in TERMINAL:
                    if result.get('progress')!=previous:print(json.dumps({'job_id':result['id'],'status':result['status'],'progress':result.get('progress')}),file=sys.stderr);previous=result.get('progress')
                    time.sleep(.3);result=store.job_get(result['id'])
            print(json.dumps(result,indent=2,allow_nan=False));return 1 if result['status'] in ['failed','cancelled','interrupted'] else 0
        if action=='web':
            from .server import serve
            serve(store,port=args['port']);return 0
        if action=='desktop':
            from .desktop import launch
            launch(store);return 0
        if action=='mcp':
            from .mcp import serve
            serve(store);return 0
        if action=='estimate':args={'parameters':json.loads(Path(args['parameters_file']).read_text())}
        method={'projects':'list_projects','jobs':'list_jobs','create':'create_project','import':'import_dataset','llm-recipe':'llm_recipe','workflow-jobs':'workflow_jobs','job':'job_get','logs':'job_logs','cancel':'job_cancel'}.get(action,action)
        result=getattr(store,method)(**args);print(json.dumps(result,indent=2,allow_nan=False));return 0
    except (ValueError,TypeError,OSError,RuntimeError) as exc:
        print(json.dumps({'error':str(exc)}));return 1
