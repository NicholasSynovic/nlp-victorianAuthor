from argparse import Namespace

from pandas import DataFrame
from progress.bar import Bar
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from nlp_final.data import dataHandler


def main(args: Namespace) -> None:
    df: DataFrame = dataHandler.loadData(path=args.training_dataset)
    dfData: list = dataHandler.splitData(df)


def trainMultinomialNaiveBayes(trainingData: DataFrame) -> None:
    pipeline: Pipeline = make_pipeline(
        CountVectorizer(), StandardScaler(), MultinomialNB()
    )

    with Bar(
        "Training Multinomial Naive Bayes classifier with CountVectorizer feature extraction..."
    ) as bar:
        pass
