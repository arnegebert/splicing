#!/bin/bash
for configname in configs/*.json; do
  python train.py -c $configname
done