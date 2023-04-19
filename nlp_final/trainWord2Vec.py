from pathlib import Path
from typing import Tuple

import dataHandler
from gensim.models import Word2Vec
from pandas import DataFrame, Series

dataPath: Path = Path("../dataset/training.csv")
vectorizerPath: Path = Path("../models/word2vec.gensim").resolve()


def trainWord2Vec(x: Series) -> Word2Vec:
    w2v: Word2Vec = Word2Vec(sentences=x, vector_size=500, seed=42)
    return w2v


def main() -> None:
    print("Splitting data...")
    df: DataFrame = dataHandler.loadData(path=dataPath)
    splits: Tuple[DataFrame] = dataHandler.splitData(df=df)
    trainingSplit: DataFrame = splits.train

    print("Training Word2Vec vectorizer...")
    v: Word2Vec = trainWord2Vec(x=trainingSplit["text"])

    print(f"Saving vectorizer to {vectorizerPath.__str__()}")
    v.save(vectorizerPath.__str__())


if __name__ == "__main__":
    main()
