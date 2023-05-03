#!/bin/bash

cd models

wget https://huggingface.co/NicholasSynovic/VEAA-Models/resolve/main/CountVectorizer.joblib

wget https://huggingface.co/NicholasSynovic/VEAA-Models/resolve/main/MultinomialNB_CV.joblib

wget https://huggingface.co/NicholasSynovic/VEAA-Models/resolve/main/MultinomialNB_TFIDF.joblib

wget https://huggingface.co/NicholasSynovic/VEAA-Models/resolve/main/TfidfVectorizer.joblib

wget https://huggingface.co/NicholasSynovic/VEAA-Models/resolve/main/Tokenizer_CNN.joblib

wget https://huggingface.co/NicholasSynovic/VEAA-Models/resolve/main/Tokenizer_RNN.joblib

wget https://huggingface.co/NicholasSynovic/VEAA-Models/resolve/main/veaaCNN.h5

wget https://huggingface.co/NicholasSynovic/VEAA-Models/resolve/main/veaaRNN.h5
