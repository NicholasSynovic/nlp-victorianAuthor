#!/bin/bash

cd nlp_final

echo "python3.10 trainFastText.py"
python3.10 trainFastText.py
echo "==="

echo "python3.10 trainTFIDF.py"
python3.10 trainTFIDF.py
echo "==="

echo "python3.10 trainWord2Vec.py"
python3.10 trainWord2Vec.py
echo "==="
