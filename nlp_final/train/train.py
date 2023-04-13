from argparse import Namespace
from typing import List

from numpy import ndarray
from pandas import DataFrame

from nlp_final.data import dataHandler
from nlp_final.train import vectorizer


def main(args: Namespace) -> None:
    df: DataFrame = dataHandler.loadData(path=args.training_dataset)
    data: List[DataFrame] = dataHandler.splitData(df)

    x: ndarray = data[0]["text"].to_numpy()
    y: ndarray = data[0]["author"].to_numpy()

    validationX: ndarray = data[1]["text"].to_numpy()
    validationY: ndarray = data[1]["author"].to_numpy()

    match args.modelType:
        case "vectorizer":
            vectorizer.main(args=args, x=x)
        case "classifier":
            pass
