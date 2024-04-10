#!/bin/bash

. env.sh

if [ $# != 1 ]; then 
  echo "usage: $0 <lang>"; exit 1
fi

lang=$1

echo START `date`

mkdir -p checkpoints/$lang

set -e -u -o pipefail

python recasepunc.py --lang=$lang train data/$lang-100M.train.x data/$lang-100M.train.y data/$lang-100M.valid.x data/$lang-100M.valid.y checkpoints/$lang

echo END `date`
