#!/bin/bash

#while read FILENAME URL; do
#  wget -nc -O "$FILENAME" "$URL"
#done < BOSS_Legacy_urls.txt

cd ../images-legacy

# in parallel --> note download speed is about 425 images/minute for jobs=8 on GCP
jobs=8
xargs < ../data/legacy_urls_mini.txt -P ${jobs} -L 1 wget -q -O 
