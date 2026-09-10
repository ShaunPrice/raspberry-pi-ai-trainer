"""Build the native desktop and companion console launchers on the current OS."""
import os
import subprocess
import sys
from pathlib import Path

root=Path(__file__).resolve().parents[1]
os.environ.setdefault('PYINSTALLER_CONFIG_DIR',str(root/'build/pyinstaller-cache'))
common=[sys.executable,'-m','PyInstaller','--noconfirm','--clean','--paths',str(root),'--collect-submodules','pi_trainer','--collect-data','pi_trainer','--add-data',str(root/'pi_trainer')+':pi_trainer','--exclude-module','torch','--exclude-module','transformers','--exclude-module','onnx','--exclude-module','numpy','--exclude-module','cv2','--exclude-module','peft','--distpath',str(root/'dist/native'),'--workpath',str(root/'build/native'),'--specpath',str(root/'build')]
subprocess.run(common+['--windowed','--name','Pi Trainer',str(root/'packaging/entry.py')],check=True,cwd=root)
subprocess.run(common+['--console','--name','pi-trainer',str(root/'packaging/entry.py')],check=True,cwd=root)
