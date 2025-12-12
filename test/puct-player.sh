#!/bin/bash

port="$1"

cd ..

python puct-player.py --port=$port --model=cai10000CP-32-9-LeaRate000-vx32-vy25-batchsize512.bin