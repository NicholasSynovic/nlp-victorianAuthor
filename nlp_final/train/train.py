from argparse import Namespace
from itertools import combinations
from typing import Any, List, Tuple

from joblib import dump
from numpy import ndarray
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, make_pipeline

from nlp_final.data import dataHandler


def main(args: Namespace) -> None:
    df: DataFrame = dataHandler.loadData(path=args.training_dataset)
    data: List[DataFrame] = dataHandler.splitData(df)

    x: ndarray = data[0]["text"].to_numpy()
    y: ndarray = data[0]["author"].to_numpy()

    validationX: ndarray = data[1]["text"].to_numpy()
    validationY: ndarray = data[1]["author"].to_numpy()

    trainMultinomialNaiveBayes(
        x=x, y=y, validationX=validationX, validationY=validationY
    )


def trainMultinomialNaiveBayes(
    x: ndarray, y: ndarray, validationX: ndarray, validationY: ndarray
) -> None:
    ngramRanges: List[Tuple[int, int]] = list(combinations(iterable=range(1, 11), r=2))

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
        "multinomialnb__alpha": [1.0, 0, 0.5, 1.5, 2.0, 0.1],
        "multinomialnb__fit_prior": [True, False],
        "multinomialnb__force_alpha": [True, False],
    }

    pipeline: Pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())

    gsvc: GridSearchCV = GridSearchCV(estimator=pipeline, param_grid=parameters)

    gsvc.fit(X=x, y=y)

    model: Pipeline = gsvc.best_estimator_
    score: float = model.score(X=validationX, y=validationY)

    print(f"Best model score: {score}")
    dump(value=model, filename="multinomialNB.joblib")
