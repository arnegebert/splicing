#!/bin/bash
RUNID='final'
#
#for configname in configs/*.json; do
for configname in $(find configs/ -name '*.json' ! -wholename 'configs/config.json'); do
  echo "--------------------------------------------------------"
  echo "Executing experiment $configname" ;
  trap 'kill -TERM $PID' TERM INT
  python train.py -rid $RUNID -c $configname
  PID=$!
  wait $PID
  trap - TERM INT
  wait $PID
  echo "--------------------------------------------------------"
done