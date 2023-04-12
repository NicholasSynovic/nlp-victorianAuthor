from argparse import Namespace
from pathlib import Path

from pandas import DataFrame

from nlp_final.data import data


def main(args: Namespace) -> None:
    df: DataFrame = data.loadData(path=args.training_dataset)
