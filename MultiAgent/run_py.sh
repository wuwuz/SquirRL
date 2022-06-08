#!/bin/bash

# make sure the script will use your Python installation, 
# and the working directory as its home location
#export PYTHONPATH=/afs/ece/usr/charlieh/.local/lib/python3.6
export HOME=/tmp/charlieh

# run your script
/afs/ece/usr/charlieh/venv/bin/python /afs/ece/usr/charlieh/SquirRL/MultiAgent/bilinear.py --het1 $1 --het2 $2 --eta $3 --etac $4 --tau $5

