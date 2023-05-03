from pathlib import Path

import numpy
import pandas
from joblib import dump, load
from numpy import ndarray
from pandas import DataFrame, Series
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

TRAINING_DATA: Path = Path("../../dataset/training.csv")
TESTING_DATA: Path = Path("../../dataset/testing.csv")

numpy.random.seed(seed=42)


def createTFIDF(documents: Series) -> TfidfVectorizer:
    print("Creating TfidfVectorizer...")
    tfidf: TfidfVectorizer = TfidfVectorizer(
        strip_accents="unicode", lowercase=True, ngram_range=(1, 2), analyzer="word"
    )
    tfidf.fit(raw_documents=documents)
    dump(value=tfidf, filename="../../models/TfidfVectorizer.joblib")
    print("Created TfidfVectorizer ✅")
    return tfidf


def loadTFIDF() -> TfidfVectorizer:
    return load(filename="../../models/TfidfVectorizer.joblib")


def transformData(tfidf: TfidfVectorizer, documents: Series) -> spmatrix:
    print("Transforming data...")
    data: spmatrix = tfidf.transform(raw_documents=documents)
    print("Transformed data ✅")
    return data


def trainModel(x: spmatrix, y: Series) -> MultinomialNB:
    print("Training Multinomial Naive Bayes Model...")
    mnnb: MultinomialNB = MultinomialNB(force_alpha=True)
    mnnb.fit(X=x, y=y)
    dump(value=mnnb, filename="../../models/MultinomialNB_TFIDF.joblib")
    print("Trained Multinomial Naive Bayes Model ✅")
    return mnnb


def loadModel() -> MultinomialNB:
    return load(filename="../../models/MultinomialNB_TFIDF.joblib")


def evaluateModel(x: ndarray, y: Series, mnnb: MultinomialNB) -> None:
    prediction: ndarray = mnnb.predict(X=x)
    print(
        classification_report(
            y_true=y,
            y_pred=prediction,
            digits=16,
        )
    )


def main() -> None:
    trainingDF: DataFrame = pandas.read_csv(filepath_or_buffer=TRAINING_DATA)
    testingDF: DataFrame = pandas.read_csv(filepath_or_buffer=TESTING_DATA)

    xTrain, xValidation, yTrain, yValidation = train_test_split(
        trainingDF["text"],
        trainingDF["author"],
        test_size=0.15,
        train_size=0.85,
        random_state=42,
        shuffle=True,
    )

    tfidf: TfidfVectorizer = loadTFIDF()

    # Uncomment out the below section to train a vectorizer from scratch
    # tfidf: TfidfVectorizer = createTFIDF(documents=xTrain)

    xTrain: ndarray = transformData(tfidf=tfidf, documents=xTrain)
    xValidation: ndarray = transformData(tfidf=tfidf, documents=xValidation)
    xTest: ndarray = transformData(tfidf=tfidf, documents=testingDF["text"])
    yTest: Series = testingDF["author"]

    mnnb: MultinomialNB = loadModel()

    # Uncomment out the below section to train a model from scratch
    # mnnb: MultinomialNB = trainModel(x=xTrain, y=yTrain)

    evaluateModel(x=xTest, y=yTest, mnnb=mnnb)


if __name__ == "__main__":
    main()
