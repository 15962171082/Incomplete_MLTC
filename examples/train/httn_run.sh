#!/bin/bash

set -vx

echo "$(git log -1)"
# git diff

export TOKENIZERS_PARALLELISM=false
python -u examples/train/httn_run.py -c "examples/train/httn_config.yaml"