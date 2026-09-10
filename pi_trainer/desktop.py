"""Native Tk workbench; all actions use the same Store as HTTP/CLI/MCP."""
from __future__ import annotations
import json
import queue
import threading
import os
import sys
from pathlib import Path

def launch(store):
    try:
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
    except ImportError as exc:
        raise RuntimeError('Desktop requires Python with Tk support. Use the web interface or install a Tk-enabled Python distribution.') from exc
    # Relocatable Python distributions can ship Tcl/Tk beside Python while their
    # compiled fallback still points to the builder's directory.
    for variable, folder, marker in [('TCL_LIBRARY',f'tcl{tk.TclVersion}','init.tcl'),('TK_LIBRARY',f'tk{tk.TkVersion}','tk.tcl')]:
        candidate=Path(sys.base_prefix)/'lib'/folder
        if (candidate/marker).is_file():os.environ.setdefault(variable,str(candidate))
    root = tk.Tk()
    root.title('Pi Trainer — model workbench')
    root.geometry('1050x740')
    root.minsize(780, 560)
    selected = tk.StringVar()
    name = tk.StringVar()
    target = tk.StringVar(value='hailo8l')
    task = tk.StringVar(value='vision')
    status = tk.StringVar(value='Local workbench • select configured compute environments and enrolled devices')
    frame = ttk.Frame(root, padding=22); frame.pack(fill='both', expand=True)
    ttk.Label(frame, text='Pi Trainer', font=('Helvetica', 26, 'bold')).pack(anchor='w')
    ttk.Label(frame, text='Train, compile and deploy models for Hailo-8L, Hailo-8 and Hailo-10H').pack(anchor='w', pady=(0,18))
    form = ttk.Frame(frame); form.pack(fill='x')
    for col, (label, variable, values) in enumerate([
        ('Project name', name, None), ('Accelerator', target, ['hailo8l','hailo8','hailo10h']), ('Task', task, ['vision','llm'])]):
        ttk.Label(form, text=label).grid(row=0,column=col,sticky='w')
        widget=ttk.Entry(form,textvariable=variable) if values is None else ttk.Combobox(form,textvariable=variable,values=values,state='readonly')
        widget.grid(row=1,column=col,padx=(0,12),sticky='ew');form.columnconfigure(col,weight=1)
    ttk.Label(frame, text='Selected project').pack(anchor='w',pady=(18,4))
    projects = ttk.Combobox(frame,textvariable=selected,state='readonly');projects.pack(fill='x')
    mapping = {}
    output = tk.Text(frame,wrap='word',font=('Courier',11),height=22)
    buttons = ttk.Frame(frame);buttons.pack(fill='x',pady=14)
    output.pack(fill='both',expand=True)
    ttk.Label(frame,textvariable=status,wraplength=950).pack(anchor='w',pady=(12,0))
    pending=queue.Queue(); busy=False
    def refresh():
        old=selected.get(); mapping.clear()
        for p in store.list_projects(): mapping[f"{p['name']} · {p['target']} · {p['id']}"]=p['id']
        projects['values']=list(mapping)
        if old in mapping:selected.set(old)
        elif mapping:selected.set(next(iter(mapping)))
    def project():
        if selected.get() not in mapping:raise ValueError('Create or select a project first.')
        return mapping[selected.get()]
    def run(fn):
        nonlocal busy
        if busy:return
        busy=True;status.set('Working…')
        def work():
            try:pending.put((True,fn()))
            except Exception as exc:pending.put((False,str(exc)))
        threading.Thread(target=work,daemon=True).start()
    def poll():
        nonlocal busy
        try:
            ok,result=pending.get_nowait();busy=False
            output.delete('1.0','end');output.insert('1.0',json.dumps(result,indent=2) if ok else result)
            status.set('Operation completed. Review results; a preparation bundle is not a compiled model.' if ok else 'Operation failed: '+result)
            refresh()
        except queue.Empty:pass
        root.after(100,poll)
    def create():
        n,t,k=name.get(),target.get(),task.get();run(lambda:store.create_project(n,t,k))
    def import_data(folder):
        try:pid=project()
        except ValueError as exc:messagebox.showerror('Select a project',str(exc));return
        path=filedialog.askdirectory() if folder else filedialog.askopenfilename()
        if path:run(lambda:store.import_dataset(pid,path))
    def action(method):
        try:pid=project()
        except ValueError as exc:messagebox.showerror('Select a project',str(exc));return
        run(lambda:method(pid))
    for index, (label, fn) in enumerate([('Create project',create),('Import folder',lambda:import_data(True)),('Import file',lambda:import_data(False)),('Inspect plan',lambda:action(store.plan)),('Export bundle',lambda:action(store.bundle)),('LLM tools',lambda:llm_window()),('Training & deployment',lambda:workflow_window()),('Refresh',refresh),('Raspberry Pi Imager',lambda:imager())]):
        ttk.Button(buttons,text=label,command=fn).grid(row=index//4,column=index%4,sticky='ew',padx=(0,7),pady=3)
    def imager():
        path=filedialog.askopenfilename(title="Open customised image in Raspberry Pi Imager",filetypes=[("Raw OS image","*.img")])
        if path:run(lambda:store.open_imager(path))
    def workflow_window():
        from .desktop_workflows import launch as launch_workflows
        launch_workflows(store,root)
    def llm_window():
        window=tk.Toplevel(root);window.title('LLM-Optimise tools — Pi Trainer');window.geometry('660x650')
        body=ttk.Frame(window,padding=20);body.pack(fill='both',expand=True)
        ttk.Label(body,text='Host preparation and saved-output evaluation',font=('Helvetica',16,'bold')).pack(anchor='w')
        ttk.Label(body,text='No training, inference or Hailo compilation is run here.').pack(anchor='w',pady=8)
        notebook=ttk.Notebook(body);notebook.pack(fill='both',expand=True)
        def field(parent,label,value=''):
            ttk.Label(parent,text=label).pack(anchor='w',pady=(12,2));v=tk.StringVar(value=value)
            ttk.Entry(parent,textvariable=v).pack(fill='x');return v
        recipe=ttk.Frame(notebook,padding=14);notebook.add(recipe,text='Host recipe')
        engine=field(recipe,'Engine: soup-mlx / soup-qlora / soup-stream','soup-mlx')
        model=field(recipe,'Base model path or ID');data=field(recipe,'Reviewed Alpaca JSONL path')
        def prepare():
            try:pid=project()
            except ValueError as exc:messagebox.showerror('Select project',str(exc));return
            e,m,d=engine.get(),model.get(),data.get();run(lambda:store.llm_recipe(pid,e,m,d))
        ttk.Button(recipe,text='Prepare host recipe',command=prepare).pack(anchor='w',pady=18)
        scoring=ttk.Frame(notebook,padding=14);notebook.add(scoring,text='Score outputs')
        tasks=field(scoring,'Evaluation tasks JSONL path');outputs=field(scoring,'Saved outputs JSON path')
        def score():
            t,o=tasks.get(),outputs.get();run(lambda:store.score(t,o))
        ttk.Button(scoring,text='Score saved outputs',command=score).pack(anchor='w',pady=18)
        memory=ttk.Frame(notebook,padding=14);notebook.add(memory,text='Memory estimate')
        ttk.Label(memory,text='Dense-transformer architecture parameters (JSON)').pack(anchor='w')
        parameters=tk.Text(memory,height=8,wrap='word');parameters.pack(fill='x',pady=10)
        parameters.insert('1.0','{"parameters_b":1.5,"weight_bits":4,"layers":28,"kv_heads":2,"head_dim":128,"context":2048}')
        def estimate():
            try:values=json.loads(parameters.get('1.0','end'))
            except ValueError as exc:messagebox.showerror('Invalid JSON',str(exc));return
            run(lambda:store.estimate(values))
        ttk.Button(memory,text='Estimate memory',command=estimate).pack(anchor='w')
        ttk.Label(memory,text='Estimate only; not a Hailo fit guarantee.').pack(anchor='w',pady=15)
        ttk.Label(body,text='Results appear in the main workbench window.').pack(anchor='w',pady=12)
    refresh();poll();root.mainloop()
