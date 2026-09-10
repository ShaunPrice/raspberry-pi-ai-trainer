"""One allowlist and schema set for HTTP, CLI and MCP preparation tools."""
TOOLS=[
 ('capabilities','List implemented and unavailable capabilities',{}),
 ('list_projects','List projects',{}),
 ('create_project','Create a target-specific project',{'name':'string','target':'string','task':'string'}),
 ('import_dataset','Snapshot a local dataset',{'project_id':'string','path':'string'}),
 ('datasets','Inspect dataset manifests',{'project_id':'string'}),
 ('plan','Inspect build readiness; does not execute stages',{'project_id':'string'}),
 ('list_jobs','List recent completed preparation operations',{}),
 ('bundle','Export a preparation ZIP, not a model or OS image',{'project_id':'string'}),
 ('llm_recipe','Prepare a host training recipe without running training',{'project_id':'string','engine':'string','model':'string','data':'string'}),
 ('score','Score supplied outputs using LLM-Optimise evaluators',{'tasks_path':'string','outputs_path':'string'}),
 ('estimate','Estimate dense-transformer inference memory; no Hailo fit guarantee',{'parameters':'object'})]

TOOLS += [
 ('run_job','Run an explicit training/build/deployment/image workflow',{'project_id':'string','kind':'string','spec':'object'}),
 ('workflow_jobs','List durable workflows',{}),
 ('job_get','Get workflow status and result',{'job_id':'string'}),
 ('job_cancel','Request cancellation; inspect external state before retrying',{'job_id':'string'}),
 ('job_logs','Read bounded workflow logs',{'job_id':'string'}),
 ('doctor','Discover installed local tools and dependencies',{}),
 ('annotate','Version class labels and source/session groups',{'project_id':'string','dataset_id':'string','annotations_path':'string'}),
 ('device_enroll','Enroll an existing trusted SSH alias without connecting',{'name':'string','host':'string'}),
 ('devices','List enrolled Pis',{}),
 ('worker_enroll','Register an existing trusted Linux worker',{'name':'string','host':'string','python':'string'}),
 ('workers','List registered Linux workers',{}),
 ('upload_create','Start browser dataset upload',{'project_id':'string'}),
 ('upload_file','Add a bounded file to an upload',{'project_id':'string','upload_id':'string','relative_path':'string','data_base64':'string'}),
 ('upload_finish','Snapshot an uploaded dataset',{'project_id':'string','upload_id':'string'})]

TOOLS += [('write_script','Save a project script without executing it',{'project_id':'string','name':'string','content':'string'}),('project_files','List project scripts and configs',{'project_id':'string'})]

TOOLS += [('prepare_imager','Prepare offline catalogue for a customised image',{'image_path':'string'}),('open_imager','Open official Imager without selecting or erasing storage',{'image_path':'string'})]

def schemas():
    return [{'name':name,'description':desc,'inputSchema':{'type':'object','properties':{k:{'type':v} for k,v in args.items()},'required':list(args),'additionalProperties':False}} for name,desc,args in TOOLS]

def dispatch(store,name,arguments):
    spec=next((x for x in TOOLS if x[0]==name),None)
    if spec is None:raise ValueError('Unknown operation')
    if not isinstance(arguments,dict) or set(arguments)!=set(spec[2]):raise ValueError('Operation arguments do not match the schema')
    for k,kind in spec[2].items():
        if not isinstance(arguments[k],str if kind=='string' else dict):raise ValueError(k+' must be '+kind)
    return getattr(store,name)(**arguments)
