#!/bin/bash

. env.sh

if [ $# != 1 ]; then 
  echo "usage: $0 <lang>"; exit 1
fi

lang=$1

echo START `date`

mkdir -p data

wget https://data.statmt.org/cc-100/$lang.txt.xz -O data/$lang.txt.xz

xzcat data/$lang.txt.xz | python recasepunc.py --lang=$lang preprocess 100000000 > data/$lang-100M.txt

set -e -u -o pipefail

tail -n +20000 data/$lang-100M.txt > data/$lang-100M.train.txt
head -10000 data/$lang-100M.txt > data/$lang-100M.test.txt
head -20000 data/$lang-100M.txt | tail -10000 > data/$lang-100M.valid.txt

for subset in train valid test; do
  python recasepunc.py --lang=$lang tensorize data/$lang-100M.$subset.txt data/$lang-100M.$subset.x data/$lang-100M.$subset.y
done

echo END `date`
