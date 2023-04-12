from pathlib import Path

from pandas import DataFrame

from nlp_final.data import data


def main() -> None:
    df: DataFrame = data.loadData(path=trainingDataset)
    print(df.head())
    pass
