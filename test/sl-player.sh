#!/bin/bash

port="$1"

cd ..

python sl-player.py --port=$port --model=cai10000CP-32-9-LeaRate000.bin