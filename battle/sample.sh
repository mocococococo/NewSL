#!/bin/bash

port="$1"
model="$2"
dir="$3"



if [ -z "$port" ]; then
    echo "No port provided"
    exit 1
fi

if [ -z "$model" ]; then
    echo "No model provided"
    exit 1
fi

if [ -z "$dir" ]; then
    echo "No directory provided"
else
    echo "Directory: $dir"
fi

cd "$dir"
if [ $? -ne 0 ]; then
    echo "Failed to change directory to $dir"
    exit 1
fi

model="${model}.bin"

python sample.py --port=$port --model=$model --use_gpu=True --name=$model
