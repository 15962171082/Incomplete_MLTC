#!/bin/bash

set -vx

echo "$(git log -1)"
# git diff

export TOKENIZERS_PARALLELISM=false
python -u examples/train/self_training_run.py -c "examples/train/pst_config.yaml"