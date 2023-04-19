from typing import Tuple

import dataHandler
from common import loadVectorizers, trainingDataPath
from gensim.models import FastText, Word2Vec
from joblib import dump, load
from pandas import DataFrame
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier


def main() -> None:
    print("Splitting data...")
    df: DataFrame = dataHandler.loadData(path=trainingDataPath)
    splits: Tuple[DataFrame] = dataHandler.splitData(df=df)
    trainingSplit: DataFrame = splits.train
    testingSplit: DataFrame = splits.test

    print("Loading vectorizers...")
    fasttext: FastText
    tfidf: TfidfVectorizer
    word2vec: Word2Vec
    fasttext, tfidf, word2vec = loadVectorizers()

    print("Training MLP with TF-IDF...")
    mlp: MLPClassifier = MLPClassifier()
    trainingDocuments: spmatrix = tfidf.transform(raw_documents=trainingSplit["text"])
    mlp.fit(X=trainingDocuments, y=trainingSplit["author"])

    print("Training MLP with Word2Vec...")


if __name__ == "__main__":
    print("Do not use! Will crash computer")
