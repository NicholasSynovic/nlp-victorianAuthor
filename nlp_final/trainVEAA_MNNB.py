from pathlib import Path

import numpy
import pandas
from joblib import dump, load
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

TRAINING_DATA: Path = Path("../dataset/trainingData_5-2-2023.csv")
TESTING_DATA: Path = Path("../dataset/testingData_5-2-2023.csv")

numpy.random.seed(seed=42)


def createCV(documents: Series) -> CountVectorizer:
    print("Creating CountVectorizer...")
    cv: CountVectorizer = CountVectorizer(
        strip_accents="unicode", lowercase=True, ngram_range=(1, 2), analyzer="word"
    )
    cv.fit(raw_documents=documents)
    print("Created CountVectorizer ✅")
    dump(value=cv, filename="../models/countVectorizer.joblib")
    return cv


def loadCV() -> CountVectorizer:
    return load(filename="../models/countVectorizer.joblib")


def transformData(cv: CountVectorizer, documents: Series) -> ndarray:
    print("Tranforming data...")
    data: ndarray = cv.transform(raw_documents=documents)
    print("Tranformed data ✅")
    return data


def trainModel(x: ndarray, y: Series) -> MultinomialNB:
    print("Training Multinomial Naive Bayes Model...")
    mnnb: MultinomialNB = MultinomialNB(force_alpha=True)
    mnnb.fit(X=x, y=y)
    print("Trained Multinomial Naive Bayes Model ✅")
    dump(value=mnnb, filename="../models/mnnb.joblib")
    return mnnb


def loadModel() -> MultinomialNB:
    return load(filename="../models/mnnb.joblib")


def evaluateModel(x: ndarray, y: Series, mnnb: MultinomialNB) -> None:
    prediction: ndarray = mnnb.predict(X=x)
    print("Accuracy: ", accuracy_score(y_true=y, y_pred=prediction))


def main() -> None:
    trainingDF: DataFrame = pandas.read_csv(filepath_or_buffer=TRAINING_DATA)

    xTrain, xTest, yTrain, yTest = train_test_split(
        trainingDF["text"],
        trainingDF["author"],
        test_size=0.15,
        train_size=0.85,
        random_state=42,
        shuffle=True,
    )

    cv: CountVectorizer = createCV(documents=xTrain)
    # cv: CountVectorizer = loadCV()

    xTrain: ndarray = transformData(cv=cv, documents=xTrain)
    xTest: ndarray = transformData(cv=cv, documents=xTest)

    mnnb: MultinomialNB = trainModel(x=xTrain, y=yTrain)
    # mnnb: MultinomialNB = loadModel()

    evaluateModel(x=xTest, y=yTest, mnnb=mnnb)


if __name__ == "__main__":
    main()
