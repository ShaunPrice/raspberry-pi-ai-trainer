#!/bin/zsh
set -eu
trainer_root="${0:A:h}"
export PI_TRAINER_TRAINING_PYTHON="$trainer_root/.venv-training/bin/python"
cd "$trainer_root"
exec "$trainer_root/dist/native/Pi Trainer.app/Contents/MacOS/Pi Trainer" --data-dir "$trainer_root/.trainer" desktop
