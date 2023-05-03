# Victorian Era Classifcation

> LUC COMP 329-429 Final

## Table of Contents

- [Victorian Era Classifcation](#victorian-era-classifcation)
  - [Table of Contents](#table-of-contents)
  - [Authors](#authors)
  - [About](#about)
    - [Dataset](#dataset)
  - [How to Run](#how-to-run)
    - [Setup](#setup)
    - [Baseline](#baseline)
      - [Train a Model (Baseline)](#train-a-model-baseline)
      - [Evaluate A Pre-Trained Model (Baseline)](#evaluate-a-pre-trained-model-baseline)
    - [CNN](#cnn)
      - [Train a Model (CNN)](#train-a-model-cnn)
      - [Evaluate A Pre-Trained Model (CNN)](#evaluate-a-pre-trained-model-cnn)
    - [RNN](#rnn)
      - [Train a Model (RNN)](#train-a-model-rnn)
      - [Evaluate A Pre-Trained Model (RNN)](#evaluate-a-pre-trained-model-rnn)

## Authors

Authors are listed in alphabetical order by last name:

- Giorgio Montenegro
- Nicholas Synovic
- Stefan Nelson

## About

This repository contains the source code and final paper for our Loyola
University Chicago (LUC) COMP 329-429 Natural Language Processing (NLP) Final.

Our project, in short, is to develop and evaluate neural networks for texts
written by authors of the Victorian Era.

### Dataset

The source dataset is the `Victorian Era Authorship Attribution Data Set` by
Abdulmecit Gungor. This dataset is hosted on the UCI Machine Learning Repository
[here](https://archive.ics.uci.edu/ml/datasets/Victorian+Era+Authorship+Attribution).

> CITATION:
> `GUNGOR, ABDULMECIT, Benchmarking Authorship Attribution Techniques Using Over A Thousand Books by Fifty Victorian Era Novelists, Purdue Master of Thesis, 2018-04`

We have modified this dataset for our purposes. We discuss our modifications
within our final paper. To download our modified dataset, please download it
from HuggingFace
[here](https://huggingface.co/datasets/NicholasSynovic/Modified-VEAA).

## How to Run

It is **required** that the setup instructions are followed.

### Setup

1. `git clone` this repository to your local machine
1. Run `pip install -r requirements.txt`
1. Run `./downloadDataset.bash` to download the training and testing datasets
1. Run `./downladModels.bash` to download all of the pre-trained models

### Baseline

For evaluating or training a Multinomial Naive Bayes model

#### Train a Model (Baseline)

1. Uncomment sections prepended by
   "`# Uncomment out the below section to train a [model | vectorizer] from scratch`",
   in
   [`nlp_final/baseline/trainVEAA_MNNB_CV.py`](nlp_final/baseline/trainVEAA_MNNB_CV.py)
   or
   [`nlp_final/baseline/trainVEAA_MNNB_TFIDF.py`](nlp_final/baseline/trainVEAA_MNNB_TFIDF.py)
1. Run `python3.10 nlp_final/baseline/trainVEAA_MNNB_CV.py`
1. Run `python3.10 nlp_final/baseline/trainVEAA_MNNB_TFIDF.py`

#### Evaluate A Pre-Trained Model (Baseline)

1. Run `python3.10 nlp_final/baseline/trainVEAA_MNNB_CV.py`
1. Run `python3.10 nlp_final/baseline/trainVEAA_MNNB_TFIDF.py`

### CNN

For evaluating or training a CNN

#### Train a Model (CNN)

1. Uncomment sections prepended by
   "`# Uncomment out the below section to train a [model | tokenizer] from scratch`",
   in [`nlp_final/cnn/trainVEAA_CNN.py`](nlp_final/cnn/trainVEAA_CNN.py)
1. Run `python3.10 nlp_final/cnn/trainVEAA_CNN.py`

#### Evaluate A Pre-Trained Model (CNN)

1. Run `python3.10 nlp_final/cnn/trainVEAA_CNN.py`

### RNN

For evaluating or training an LSTM RNN

#### Train a Model (RNN)

1. Uncomment sections prepended by
   "`# Uncomment out the below section to train a [model | tokenizer] from scratch`",
   in [`nlp_final/rnn/trainVEAA_RNN.py`](nlp_final/rnn/trainVEAA_RNN.py)
1. Run `python3.10 nlp_final/rnn/trainVEAA_RNN.py`

#### Evaluate A Pre-Trained Model (RNN)

1. Run `python3.10 nlp_final/rnn/trainVEAA_RNN.py`
