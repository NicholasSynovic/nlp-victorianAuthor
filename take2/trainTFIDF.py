from pathlib import Path
from typing import Tuple

import dataHandler
from joblib import dump
from pandas import DataFrame, Series
from sklearn.feature_extraction.text import TfidfVectorizer

dataPath: Path = Path("../dataset/training.csv")
vectorizerPath: Path = Path("../models/tfidf.joblib").resolve()


def trainTFIDF(x: Series) -> TfidfVectorizer:
    tfidf: TfidfVectorizer = TfidfVectorizer(
        decode_error="ignore", lowercase=False, ngram_range=(1, 3), norm="l2"
    )
    tfidf.fit(raw_documents=x)
    return tfidf


def main() -> None:
    print("Splitting data...")
    df: DataFrame = dataHandler.loadData(path=dataPath)
    splits: Tuple[DataFrame] = dataHandler.splitData(df=df)
    trainingSplit: DataFrame = splits.train

    print("Training TF-IDF vectorizer...")

    v: TfidfVectorizer = trainTFIDF(x=trainingSplit["text"])

    print(f"Saving vectorizer to {vectorizerPath.__str__()}")
    dump(value=v, filename=vectorizerPath)


if __name__ == "__main__":
    main()
