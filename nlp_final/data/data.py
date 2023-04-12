from pathlib import Path

import pandas
from pandas import DataFrame


def loadData(path: Path) -> DataFrame:
    df: DataFrame = pandas.read_csv(filepath_or_buffer=path.absolute())
    return df
