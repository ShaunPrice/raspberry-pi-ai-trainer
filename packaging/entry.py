"""Native/console distribution launcher, including detached worker mode."""
from pathlib import Path
import os
import sys

def default_root():
    if os.environ.get('PI_TRAINER_DATA_DIR'):return Path(os.environ['PI_TRAINER_DATA_DIR'])
    if sys.platform=='darwin':return Path.home()/'Library/Application Support/Pi Trainer'
    if os.name=='nt':return Path(os.environ.get('LOCALAPPDATA',str(Path.home())))/'Pi Trainer'
    return Path(os.environ.get('XDG_DATA_HOME',str(Path.home()/'.local/share')))/'pi-trainer'

def main():
    from pi_trainer.cli import main as cli
    if sys.argv[1:2]==['--worker']:
        from pi_trainer.jobs import worker_loop
        from pi_trainer.core import Store
        worker_loop(Store(sys.argv[2]));return 0
    args=sys.argv[1:]
    if not args:args=['desktop']
    if '--data-dir' not in args:args=['--data-dir',str(default_root()),*args]
    return cli(args)
if __name__=='__main__':raise SystemExit(main())
