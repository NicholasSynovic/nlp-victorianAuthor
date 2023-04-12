from argparse import Namespace
from itertools import combinations
from typing import Any, List, Tuple

from numpy import ndarray
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, make_pipeline

from nlp_final.data import dataHandler


def main(args: Namespace) -> None:
    df: DataFrame = dataHandler.loadData(path=args.training_dataset)

    dfData: list = dataHandler.splitData(df)

    trainMultinomialNaiveBayes(trainingData=dfData[0])


def trainMultinomialNaiveBayes(trainingData: DataFrame) -> None:
    ngramRanges: List[Tuple[int, int]] = list(combinations(iterable=range(1, 11), r=2))

    x: ndarray = trainingData["text"].to_numpy()
    y: ndarray = trainingData["author"].to_numpy()

    parameters: dict[str, List[Any]] = {
        "tfidfvectorizer__analyzer": ["word", "char"],
        "tfidfvectorizer__binary": [True, False],
        "tfidfvectorizer__decode_error": ["ignore"],
        "tfidfvectorizer__lowercase": [True, False],
        "tfidfvectorizer__ngram_range": ngramRanges,
        "tfidfvectorizer__norm": ["l1", "l2"],
        "tfidfvectorizer__smooth_idf": [True, False],
        "tfidfvectorizer__stop_words": ["english", None],
        "tfidfvectorizer__strip_accents": ["ascii", "unicode", None],
        "multinomialnb__alpha": [1.0, 0],
        "multinomialnb__fit_prior": [True, False],
        "multinomialnb__force_alpha": [True, False],
    }

    pipeline: Pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())
    gsvc: GridSearchCV = GridSearchCV(estimator=pipeline, param_grid=parameters)

    gsvc.fit(X=x, y=y)
