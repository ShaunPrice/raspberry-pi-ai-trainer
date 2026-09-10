"""Native workflow controls over the shared durable-job Store API."""
from __future__ import annotations
import json
import queue
import sys
import threading
from .runtime import suggested_python

KINDS = ('train','compile','package','image','deploy','benchmark','rollback','capture','export_dataset','optimise')


def template(kind, task='vision'):
    """Independent editable examples, never submitted automatically."""
    values={
        'train':{'engine':'pytorch','python':suggested_python(),'epochs':2,'image_size':32,'batch_size':8},
        'compile':{'model_path':'','calibration_path':'','python':suggested_python()},
        'package':{'model_path':'','runtime_version':'','scripts_dir':'','entrypoint':None},
        'image':{'base_image':'','bundle_path':'','install_service':False},
        'deploy':{'device_id':'','bundle_path':'','activate':True},
        'benchmark':{'device_id':'','action':'benchmark','mode':task},
        'rollback':{'device_id':''},
        'capture':{'seconds':5,'fps':1,'camera_index':0,'python':suggested_python()},
        'export_dataset':{},
        'optimise':{'engine':'pytorch','python':suggested_python(),'search':{'image_size':[32,64],'epochs':[1]},'max_trials':2,'min_quality':0.5},
    }
    if kind not in values:raise ValueError('Unknown workflow kind')
    if task=='llm':
        values['train']={'engine':'transformers-peft','python':suggested_python(),'model':'','epochs':1,'rank':8,'batch_size':1,'device':'cpu','allow_download':False}
        values['compile']={'model_path':'','recipe_executable':''}
        values['optimise']={'engine':'transformers-peft','python':suggested_python(),'model':'','search':{'rank':[4,8],'epochs':[1]},'max_trials':2,'min_quality':0.01}
        values['benchmark']={'device_id':'','action':'benchmark','mode':'llm_direct','max_tokens':64}
    return values[kind]


def parse_spec(text):
    value=json.loads(text,parse_constant=lambda v: (_ for _ in ()).throw(ValueError('Nonfinite JSON value: '+v)))
    if not isinstance(value,dict):raise ValueError('The specification must be a JSON object')
    return value


def launch(store, parent):
    """Open a nonmodal native dialog; Store calls stay off the Tk thread."""
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    window=tk.Toplevel(parent);window.title('Pi Trainer — training and deployment');window.geometry('1120x830');window.minsize(850,650)
    body=ttk.Frame(window,padding=16);body.pack(fill='both',expand=True)
    ttk.Label(body,text='Training and deployment',font=('Helvetica',21,'bold')).pack(anchor='w')
    ttk.Label(body,text='Host training • Hailo compilation • image customisation • network testing').pack(anchor='w',pady=(2,12))
    status=tk.StringVar(value='Loading projects and jobs…')
    notebook=ttk.Notebook(body);notebook.pack(fill='both',expand=True)
    flow=ttk.Frame(notebook,padding=10);registry=ttk.Frame(notebook,padding=10);diagnostics=ttk.Frame(notebook,padding=10)
    notebook.add(flow,text='Workflows');notebook.add(registry,text='Devices and workers');notebook.add(diagnostics,text='Dependencies')
    ttk.Label(body,textvariable=status,wraplength=1060).pack(fill='x',pady=(10,0))
    pending=queue.Queue();inflight=set();state={'closed':False,'projects':{},'devices':{},'workers':{},'selected_job':None}
    def submit(key, function, callback, silent=False):
        if key in inflight or state['closed']:return
        inflight.add(key)
        def work():
            try:pending.put((key,True,function(),callback,silent))
            except Exception as exc:pending.put((key,False,str(exc),callback,silent))
        threading.Thread(target=work,daemon=True).start()
    def show(widget,value):
        text=value if isinstance(value,str) else json.dumps(value,indent=2,default=str)
        old=widget.get('1.0','end-1c')
        if old==text:return
        position=widget.yview();widget.configure(state='normal');widget.delete('1.0','end');widget.insert('1.0',text);widget.configure(state='disabled')
        if position:widget.yview_moveto(position[0])
    def text_area(parent,height=10):
        frame=ttk.Frame(parent);frame.pack(fill='both',expand=True)
        widget=tk.Text(frame,height=height,wrap='word',font=('Courier',11));bar=ttk.Scrollbar(frame,orient='vertical',command=widget.yview)
        widget.configure(yscrollcommand=bar.set);bar.pack(side='right',fill='y');widget.pack(side='left',fill='both',expand=True)
        return widget
    controls=ttk.Frame(flow);controls.pack(fill='x')
    project=tk.StringVar();kind=tk.StringVar(value='train')
    ttk.Label(controls,text='Project').grid(row=0,column=0,sticky='w');ttk.Label(controls,text='Workflow').grid(row=0,column=1,sticky='w')
    project_box=ttk.Combobox(controls,textvariable=project,state='readonly');project_box.grid(row=1,column=0,sticky='ew',padx=(0,10))
    kind_box=ttk.Combobox(controls,textvariable=kind,values=KINDS,state='readonly',width=20);kind_box.grid(row=1,column=1,sticky='ew')
    controls.columnconfigure(0,weight=1)
    ttk.Label(flow,text='Job specification (JSON). Choose a template, fill paths and IDs, then run explicitly.').pack(anchor='w',pady=(10,4))
    spec=text_area(flow,9)
    def read_spec():return parse_spec(spec.get('1.0','end'))
    def write_spec(value):spec.delete('1.0','end');spec.insert('1.0',json.dumps(value,indent=2))
    def project_value():
        value=state['projects'].get(project.get())
        if not value:raise ValueError('Create a project in the main workbench and refresh, then select it')
        return value
    def reset_template(event=None):
        selected=state['projects'].get(project.get(),{})
        write_spec(template(kind.get(),selected.get('task','vision')))
        status.set('Template loaded. Empty paths/runtime versions must be supplied; no job has run.')
    kind_box.bind('<<ComboboxSelected>>',reset_template)
    # Changing a project does not erase an operator-edited specification.
    commands=ttk.Frame(flow);commands.pack(fill='x',pady=7)
    def submitted(result):
        state['selected_job']=result['id'];status.set('Job '+result['id']+' queued. Results and failures appear below.');refresh_jobs()
    def run():
        try:pid=project_value()['id'];operation=kind.get();value=read_spec()
        except Exception as exc:messagebox.showerror('Job specification',str(exc),parent=window);return
        submit('run',lambda:store.run_job(pid,operation,value),submitted)
    def select_path(directory=False):
        try:value=read_spec()
        except ValueError as exc:messagebox.showerror('Job specification',str(exc),parent=window);return
        field=path_field.get().strip()
        if not field:status.set('Select the JSON field that should receive the path.');return
        chosen=filedialog.askdirectory(parent=window) if directory else filedialog.askopenfilename(parent=window)
        if chosen:value[field]=chosen;write_spec(value)
    ttk.Button(commands,text='Load template',command=reset_template).pack(side='left')
    ttk.Button(commands,text='Run job',command=run).pack(side='left',padx=6)
    path_field=tk.StringVar(value='model_path')
    ttk.Combobox(commands,textvariable=path_field,values=('model_path','model','base_image','bundle_path','calibration_path','scripts_dir','python','recipe_executable'),width=20).pack(side='left',padx=8)
    ttk.Button(commands,text='Choose file',command=lambda:select_path(False)).pack(side='left')
    ttk.Button(commands,text='Choose folder',command=lambda:select_path(True)).pack(side='left',padx=6)
    resources=ttk.Frame(flow);resources.pack(fill='x',pady=(0,7))
    device=tk.StringVar();worker=tk.StringVar()
    device_box=ttk.Combobox(resources,textvariable=device,state='readonly',width=25);device_box.pack(side='left')
    worker_box=ttk.Combobox(resources,textvariable=worker,state='readonly',width=25);worker_box.pack(side='left',padx=8)
    def insert_resource():
        try:
            value=read_spec()
            if device.get() in state['devices']:value['device_id']=state['devices'][device.get()]['id']
            if worker.get() in state['workers']:value['worker_id']=state['workers'][worker.get()]['id']
            else:value.pop('worker_id',None)
            write_spec(value)
        except ValueError as exc:messagebox.showerror('Job specification',str(exc),parent=window)
    ttk.Button(resources,text='Use selected device / worker',command=insert_resource).pack(side='left')
    job_header=ttk.Frame(flow);job_header.pack(fill='x')
    ttk.Label(job_header,text='Recent jobs').pack(side='left')
    table=ttk.Treeview(flow,columns=('kind','status','stage'),show='tree headings',height=5)
    table.heading('#0',text='Job ID');table.column('#0',width=260)
    for column in ('kind','status','stage'):table.heading(column,text=column.title());table.column(column,width=110)
    table.pack(fill='x',pady=5)
    details_tabs=ttk.Notebook(flow);details_tabs.pack(fill='both',expand=True)
    detail_frame=ttk.Frame(details_tabs);log_frame=ttk.Frame(details_tabs)
    details_tabs.add(detail_frame,text='Progress and result');details_tabs.add(log_frame,text='Logs')
    detail=text_area(detail_frame,8);logs=text_area(log_frame,8)
    def display_jobs(values):
        selected=state['selected_job'];existing=set(table.get_children())
        for job in values:
            jid=job['id'];progress=job.get('progress') or {}
            stage=progress.get('stage',progress.get('message','')) if isinstance(progress,dict) else str(progress)
            row=(job['kind'],job['status'],stage)
            if jid in existing:table.item(jid,values=row);existing.remove(jid)
            else:table.insert('', 'end',iid=jid,text=jid,values=row)
        for jid in existing:table.delete(jid)
        if selected and table.exists(selected):table.selection_set(selected)
    def refresh_jobs():submit('jobs',store.workflow_jobs,display_jobs,True)
    def display_detail(value):
        jid,job,log=value
        if state['selected_job']==jid:show(detail,job);show(logs,log)
    def refresh_detail():
        jid=state['selected_job']
        if jid:submit('detail',lambda:(jid,store.job_get(jid),store.job_logs(jid)),display_detail,True)
    def selected_job(event=None):
        choices=table.selection()
        if choices:state['selected_job']=choices[0];refresh_detail()
    table.bind('<<TreeviewSelect>>',selected_job)
    def cancel_job():
        jid=state['selected_job']
        if jid:submit('cancel',lambda:store.job_cancel(jid),lambda result:(status.set('Cancellation requested for '+jid),refresh_jobs(),refresh_detail()))
        else:status.set('Select a job first.')
    ttk.Button(job_header,text='Refresh jobs',command=refresh_jobs).pack(side='right')
    ttk.Button(job_header,text='Cancel selected',command=cancel_job).pack(side='right',padx=8)
    ttk.Label(registry,text='Enroll existing trusted SSH aliases. Configure keys and known_hosts outside this dialog.').pack(anchor='w')
    enroll_form=ttk.Frame(registry);enroll_form.pack(fill='x',pady=12)
    enrollment_kind=tk.StringVar(value='device');entry_name=tk.StringVar();entry_host=tk.StringVar();entry_python=tk.StringVar(value='python3')
    for col,(label,variable) in enumerate((('Name',entry_name),('SSH alias',entry_host),('Worker Python',entry_python))):
        ttk.Label(enroll_form,text=label).grid(row=0,column=col,sticky='w')
        ttk.Entry(enroll_form,textvariable=variable).grid(row=1,column=col,sticky='ew',padx=(0,10));enroll_form.columnconfigure(col,weight=1)
    ttk.Combobox(enroll_form,textvariable=enrollment_kind,values=('device','worker'),state='readonly',width=10).grid(row=1,column=3)
    registry_output=text_area(registry,16)
    def registry_result(result):
        ds,ws=result
        state['devices']={d['name']+' · '+d['id']:d for d in ds};state['workers']={w['name']+' · '+w['id']:w for w in ws}
        device_box['values']=list(state['devices']);worker_box['values']=['Local host',*state['workers']]
        if not worker.get():worker.set('Local host')
        show(registry_output,{'devices':ds,'workers':ws})
    def refresh_registry():submit('registry',lambda:(store.devices(),store.workers()),registry_result,True)
    def enroll():
        name,host,python=entry_name.get(),entry_host.get(),entry_python.get();category=enrollment_kind.get()
        fn=(lambda:store.device_enroll(name,host)) if category=='device' else (lambda:store.worker_enroll(name,host,python))
        submit('enroll',fn,lambda result:(status.set('Enrolled '+name+'; connection not yet verified.'),refresh_registry()))
    ttk.Button(enroll_form,text='Enroll',command=enroll).grid(row=1,column=4,padx=8)
    ttk.Button(registry,text='Refresh registry',command=refresh_registry).pack(anchor='w',pady=8)
    ttk.Label(diagnostics,text='Installed-tool discovery only. Hailo SDK licensing, supported host architecture, and actual device execution are separate checks.',wraplength=1000).pack(anchor='w')
    doctor_output=text_area(diagnostics,22)
    def doctor():submit('doctor',store.doctor,lambda value:show(doctor_output,value))
    ttk.Button(diagnostics,text='Inspect dependencies',command=doctor).pack(anchor='w',pady=8)
    def project_result(values):
        old=project.get();state['projects']={p['name']+' · '+p['target']+' · '+p['id']:p for p in values}
        project_box['values']=list(state['projects'])
        if old not in state['projects'] and state['projects']:project.set(next(iter(state['projects'])))
    def refresh_projects():submit('projects',store.list_projects,project_result,True)
    ttk.Button(controls,text='Refresh projects',command=refresh_projects).grid(row=1,column=2,padx=8)
    def poll():
        if state['closed']:return
        try:
            while True:
                key,ok,result,callback,silent=pending.get_nowait();inflight.discard(key)
                if ok:
                    try:callback(result)
                    except Exception as exc:status.set('Display failed: '+str(exc))
                else:
                    status.set(key+' failed: '+result)
                    if not silent:messagebox.showerror('Operation failed',result,parent=window)
        except queue.Empty:pass
        window.after(100,poll)
    def periodic():
        if state['closed']:return
        refresh_jobs();refresh_detail();window.after(2000,periodic)
    def close():state['closed']=True;window.destroy()
    window.protocol('WM_DELETE_WINDOW',close)
    window.bind('<Destroy>',lambda event:state.update(closed=True) if event.widget is window else None)
    reset_template();refresh_projects();refresh_registry();doctor();poll();periodic()
    return window
