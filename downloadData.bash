#!/bin/bash

wget -O data.zip https://archive.ics.uci.edu/ml/machine-learning-databases/00454/dataset.zip

unzip data.zip
rm data.zip

iconv dataset/Gungor_2018_VictorianAuthorAttribution_data-train.csv -c -t UTF-8 -o dataset/training.csv
iconv dataset/Gungor_2018_VictorianAuthorAttribution_data.csv -c -t UTF-8 -o dataset/testing.csv
