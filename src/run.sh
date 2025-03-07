#!/bin/bash

clear

WORK_DIR=$(pwd)
WORKSPACE=$WORK_DIR
FILE=$WORKSPACE/src/main.py # file to run
PRINT=$WORKSPACE/src/run-console.log
PYTHON=/home/geri/anaconda3/envs/ppo/bin/python # remember: /bin/python

export PATH="$WORKSPACE/:$PATH" # add new path to the PATH variable
export PYTHONPATH="$WORKSPACE:$PYTHONPATH"

$PYTHON -u $FILE \
    --host localhost \
    --port 5070 \
    --save $WORKSPACE/output \
    2>&1 | tee >(tail -n 1000 > $PRINT)