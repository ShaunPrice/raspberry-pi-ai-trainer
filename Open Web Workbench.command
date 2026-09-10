#!/bin/zsh
set -eu
trainer_root="${0:A:h}"
export PI_TRAINER_TRAINING_PYTHON="$trainer_root/.venv-training/bin/python"
cd "$trainer_root"
open http://127.0.0.1:8876
exec "$trainer_root/dist/native/pi-trainer/pi-trainer" --data-dir "$trainer_root/.trainer" web --port 8876
