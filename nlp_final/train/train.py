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
from nlp_final.train import vectorizer

patch_sklearn()


def main(args: Namespace) -> None:
    df: DataFrame = dataHandler.loadData(path=args.training_dataset)
    data: List[DataFrame] = dataHandler.splitData(df)

    x: ndarray = data[0]["text"].to_numpy()
    y: ndarray = data[0]["author"].to_numpy()

    validationX: ndarray = data[1]["text"].to_numpy()
    validationY: ndarray = data[1]["author"].to_numpy()

    match args.modelType:
        case "vectorizer":
            vectorizer.main(args=args)
        case "classifier":
            pass
