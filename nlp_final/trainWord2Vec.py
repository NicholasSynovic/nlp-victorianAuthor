from typing import Tuple

import dataHandler
from common import trainingDataPath, word2vecPath
from gensim.models import Word2Vec
from pandas import DataFrame, Series


def trainWord2Vec(x: Series) -> Word2Vec:
    w2v: Word2Vec = Word2Vec(sentences=x, vector_size=500, seed=42)
    return w2v


def main() -> None:
    print("Splitting data...")
    df: DataFrame = dataHandler.loadData(path=trainingDataPath)
    splits: Tuple[DataFrame] = dataHandler.splitData(df=df)
    trainingSplit: DataFrame = splits.train

    print("Training Word2Vec vectorizer...")
    v: Word2Vec = trainWord2Vec(x=trainingSplit["text"])

    print(f"Saving vectorizer to {word2vecPath.__str__()}")
    v.save(word2vecPath.__str__())


if __name__ == "__main__":
    main()
