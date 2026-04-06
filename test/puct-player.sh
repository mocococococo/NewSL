#!/bin/bash

port="$1"

cd ..

python puct-player.py --port=$port --model=js20000CP-32-9-LeaRate1000-vx32-vy25-batchsize1024.bin #> test/sample2.log