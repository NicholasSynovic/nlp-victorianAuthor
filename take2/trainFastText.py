from pathlib import Path
from typing import Tuple

import dataHandler
from gensim.models import FastText
from pandas import DataFrame, Series

dataPath: Path = Path("../dataset/training.csv")
vectorizerPath: Path = Path("../models/fasttext.gensim").resolve()


def trainFastText(x: Series) -> FastText:
    ft: FastText = FastText(sentences=x, vector_size=500, seed=42)
    return ft


def main() -> None:
    print("Splitting data...")
    df: DataFrame = dataHandler.loadData(path=dataPath)
    splits: Tuple[DataFrame] = dataHandler.splitData(df=df)
    trainingSplit: DataFrame = splits.train

    print("Training FastText vectorizer...")
    v: FastText = trainFastText()(x=trainingSplit["text"])

    print(f"Saving vectorizer to {vectorizerPath.__str__()}")
    v.save(vectorizerPath.__str__())


if __name__ == "__main__":
    main()
