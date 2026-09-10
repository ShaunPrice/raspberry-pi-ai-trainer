"""Detached local workbench job worker."""
import argparse
from .core import Store
from .jobs import worker_loop

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);a=p.parse_args();worker_loop(Store(a.data_dir))
if __name__=='__main__':main()
