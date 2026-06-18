#!/usr/bin/env bash

set -euo pipefail

chunk_start="${1:?Usage: ./aws-generate.sh <chunk_number>}"
chunk_end="${2:?Usage: ./aws-generate.sh <chunk_number>}"

source /home/ubuntu/env/bin/activate
cd /home/ubuntu/NewSL/transformer

tmux new-session -d -s "run_chunk_$chunk_start--$chunk_end" \
"python shot_generator.py --chunk_start=$chunk_start --chunk_end=$chunk_end > run_chunk_$chunk_start--$chunk_end.log 2>&1"