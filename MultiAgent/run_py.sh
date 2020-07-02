#!/bin/bash

# make sure the script will use your Python installation, 
# and the working directory as its home location
#export PYTHONPATH=/afs/ece/usr/charlieh/.local/lib/python3.6
export HOME=/tmp/charlieh

# run your script
/afs/ece/usr/charlieh/env/bin/python /afs/ece/usr/charlieh/SquirRL/MultiAgent/bitcoin_game.py --blocks $1 --alphas $2 --spy $3 --players $4 --OSM $5 --save_path $6 --extended $7 --team_spirit $8

