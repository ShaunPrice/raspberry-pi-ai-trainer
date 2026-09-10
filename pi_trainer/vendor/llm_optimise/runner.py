"""Local minimal JSON writer replacing the upstream experiment runner dependency."""
import json
from pathlib import Path

def write_json(path, data):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(data,stream,indent=2,allow_nan=False)
        stream.write('\n')
