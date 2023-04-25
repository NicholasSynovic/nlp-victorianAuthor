from collections import namedtuple
from pathlib import Path
from typing import Type

import numpy
import pandas
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split

output = namedtuple(typename="Output", field_names=["test", "train"])


def loadData(path: Path) -> DataFrame:
    df: DataFrame = pandas.read_csv(
        filepath_or_buffer=path.absolute(),
        dtype={"text": str, "author": int},
        engine="c",
        encoding="utf-8",
        encoding_errors="replace",
    )
    return df


def splitData(df: DataFrame) -> Type[tuple]:
    data: list = train_test_split(
        df, test_size=0.15, train_size=0.85, random_state=42, shuffle=True
    )
    return output(test=data[1], train=data[0])


def reduce(df: DataFrame) -> None:
    """Reduces to HuggingFace's dataset AutoTrain size constraints"""
    # https://www.statology.org/stratified-sampling-pandas/
    maxRowCount: int = 3000

    reducedDF: DataFrame = (
        df.groupby("author", group_keys=False)
        .apply(lambda x: x.sample(int(numpy.rint(maxRowCount * len(x) / len(df)))))
        .sample(frac=1)
        .reset_index(drop=True)
    )

    return reducedDF
