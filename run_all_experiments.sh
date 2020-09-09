#!/bin/bash
RUNID='final2'
#
#for configname in configs/*.json; do
for configname in $(find configs/hexevent/ -name '*.json' ! -wholename 'configs/config.json'); do
  echo "--------------------------------------------------------"
  echo "Executing experiment $configname" ;
  trap 'kill -TERM $PID' TERM INT
  python train.py -rid $RUNID -cv true -c $configname
  PID=$!
  wait $PID
  trap - TERM INT
  wait $PID
  echo "--------------------------------------------------------"
done