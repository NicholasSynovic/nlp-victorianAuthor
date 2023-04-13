from argparse import Namespace
from pathlib import Path
from pickle import load
from typing import Any, List

from gensim.models import FastText, Word2Vec
from numpy import ndarray
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, make_pipeline

from nlp_final.train.vectorizer import *


def main(args: Namespace, x: DataFrame, y: DataFrame) -> None:
    print(args.classifierVectorizerType)


def trainMultinomialNaiveBayes(
    x: ndarray, y: ndarray, outputPath: Path, vectorizer: TfidfVectorizer
) -> None:
    parameters: dict[str, List[Any]] = {
        "multinomialnb__alpha": [1.0, 0.5, 1.5],
        "multinomialnb__force_alpha": [True],
    }

    print(type(vectorizer))

    X = vectorizer.transform(raw_documents=x)

    # print(type(X))
    quit()

    pipeline: Pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())
    gscv: GridSearchCV = GridSearchCV(
        estimator=pipeline, param_grid=parameters, n_jobs=1
    )
    gscv.fit(X=x, y=y)
    model: Pipeline = gscv.best_estimator_
    dump(value=model, filename=Path(outputPath, "multinomialNB.joblib"))
