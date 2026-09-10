"""Unprivileged Docker worker entrypoint for existing image customisation."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from pi_trainer.images import inject_partition

if __name__ == '__main__':
    root=Path(sys.argv[1]).resolve()
    manifest=json.loads((root/'payload/manifest.json').read_text())
    inject_partition(root/'root.ext4',root/'payload',manifest,'--service' in sys.argv[2:])
    print(json.dumps({'status':'verified','release_id':manifest['release_id']}))
