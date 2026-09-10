"""Manual native-window smoke test; requires macOS/Windows/Linux desktop access."""
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import tkinter as tk
from tkinter import ttk
from pi_trainer.core import Store
from pi_trainer.desktop import launch

original=tk.Tk
results={}
temp=tempfile.TemporaryDirectory()
store=Store(Path(temp.name).resolve()/'state')
def descendants(widget):
    for child in widget.winfo_children():
        yield child
        yield from descendants(child)
class TestWindow(original):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.after(300,self.test_create)
    def report_callback_exception(self,exc,val,tb):
        results['error']=str(val);self.destroy()
    def test_create(self):
        entries=[w for w in descendants(self) if type(w) is ttk.Entry]
        entries[0].insert(0,'Native smoke vision')
        next(w for w in descendants(self) if isinstance(w,ttk.Button) and w.cget('text')=='Create project').invoke()
        self.after(500,self.check)
    def check(self):
        projects=store.list_projects()
        assert projects and projects[0]['name']=='Native smoke vision'
        results['native_project_creation']=True
        next(w for w in descendants(self) if isinstance(w,ttk.Button) and w.cget('text')=='LLM tools').invoke()
        self.update_idletasks()
        tabs=[w for w in descendants(self) if isinstance(w,ttk.Notebook)]
        assert tabs and len(tabs[0].tabs())==3
        results['llm_tool_window']=True
        next(w for w in descendants(self) if isinstance(w,ttk.Button) and w.cget('text')=='Training & deployment').invoke()
        self.update_idletasks()
        assert len([w for w in descendants(self) if isinstance(w,ttk.Notebook)])==3
        results['workflow_window']=True
        results['window_system']=self.tk.call('tk','windowingsystem')
        results['tk_version']=str(self.tk.call('info','patchlevel'))
        self.after(300,self.destroy)
tk.Tk=TestWindow
try:launch(store)
finally:
    tk.Tk=original;temp.cleanup()
Path(__file__).with_name('native-smoke.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results))
if 'error' in results:raise SystemExit(1)
