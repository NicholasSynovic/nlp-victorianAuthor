from typing import Tuple

import dataHandler
from common import fasttextPath, trainingDataPath
from gensim.models import FastText
from pandas import DataFrame, Series


def trainFastText(x: Series) -> FastText:
    ft: FastText = FastText(sentences=x, vector_size=500, seed=42)
    return ft


def main() -> None:
    print("Splitting data...")
    df: DataFrame = dataHandler.loadData(path=trainingDataPath)
    splits: Tuple[DataFrame] = dataHandler.splitData(df=df)
    trainingSplit: DataFrame = splits.train

    print("Training FastText vectorizer...")
    v: FastText = trainFastText(x=trainingSplit["text"])

    print(f"Saving vectorizer to {fasttextPath.__str__()}")
    v.save(fasttextPath.__str__())


if __name__ == "__main__":
    main()
