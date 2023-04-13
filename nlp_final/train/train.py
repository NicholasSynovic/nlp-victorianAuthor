from argparse import Namespace
from pathlib import Path
from typing import Any, List

from joblib import dump
from numpy import ndarray
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC
from sklearnex import patch_sklearn

from nlp_final.data import dataHandler

patch_sklearn()


def main(args: Namespace) -> None:
    df: DataFrame = dataHandler.loadData(path=args.training_dataset)
    data: List[DataFrame] = dataHandler.splitData(df)

    x: ndarray = data[0]["text"].to_numpy()
    y: ndarray = data[0]["author"].to_numpy()

    validationX: ndarray = data[1]["text"].to_numpy()
    validationY: ndarray = data[1]["author"].to_numpy()

    trainMultinomialNaiveBayes(
        x=x,
        y=y,
        validationX=validationX,
        validationY=validationY,
        outputPath=args.output,
    )


def trainMultinomialNaiveBayes(
    x: ndarray, y: ndarray, validationX: ndarray, validationY: ndarray, outputPath: Path
) -> None:
    parameters: dict[str, List[Any]] = {
        "tfidfvectorizer__decode_error": ["ignore"],
        "tfidfvectorizer__lowercase": [False, True],
        "tfidfvectorizer__ngram_range": [(1, 3)],
        "tfidfvectorizer__norm": ["l1", "l2"],
        "multinomialnb__alpha": [1.0, 0.5, 1.5],
        "multinomialnb__force_alpha": [True],
    }

    pipeline: Pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())

    gsvc: GridSearchCV = GridSearchCV(
        estimator=pipeline, param_grid=parameters, n_jobs=1
    )

    gsvc.fit(X=x, y=y)

    model: Pipeline = gsvc.best_estimator_
    score: float = model.score(X=validationX, y=validationY)

    print(f"Best model score: {score}")
    dump(value=model, filename=Path(outputPath, "multinomialNB.joblib"))
