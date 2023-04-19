from collections import namedtuple
from pathlib import Path
from typing import Type

import pandas
from pandas import DataFrame
from sklearn.model_selection import train_test_split

output = namedtuple(typename="Output", field_names=["test", "train"])


def loadData(path: Path) -> DataFrame:
    df: DataFrame = pandas.read_csv(filepath_or_buffer=path.absolute())
    return df


def splitData(df: DataFrame) -> Type[tuple]:
    data: list = train_test_split(
        df, test_size=0.15, train_size=0.85, random_state=42, shuffle=True
    )
    return output(test=data[1], train=data[0])
