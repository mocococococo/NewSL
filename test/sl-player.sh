#!/bin/bash

port="$1"

cd ..

python sl-player.py --port=$port --model=cai1000CP-32-9-LeaRate000.bin