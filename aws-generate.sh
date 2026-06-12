#!/usr/bin/env bash

set -euo pipefail

chunk="${1:?Usage: ./aws-generate.sh <chunk_number>}"

source /home/ubuntu/.venv/bin/activate
cd /home/ubuntu/NewSL/transformer

tmux new-session -d -s "run_chunk_$chunk" \
"python shot_generator.py --chunk=$chunk > run_chunk_$chunk.log 2>&1"