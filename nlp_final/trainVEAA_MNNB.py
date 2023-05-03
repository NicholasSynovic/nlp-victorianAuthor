from pathlib import Path

import numpy
import pandas
from joblib import dump, load
from numpy import ndarray
from pandas import DataFrame, Series
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score
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
    dump(value=cv, filename="../models/CountVectorizer.joblib")
    return cv


def loadCV() -> CountVectorizer:
    return load(filename="../models/CountVectorizer.joblib")


def createTFIDF(documents: Series) -> TfidfVectorizer:
    print("Creating TfidfVectorizer...")
    tfidf: TfidfVectorizer = TfidfVectorizer(
        strip_accents="unicode", lowercase=True, ngram_range=(1, 2), analyzer="word"
    )
    tfidf.fit(raw_documents=documents)
    print("Created TfidfVectorizer ✅")
    dump(value=tfidf, filename="../models/TfidfVectorizer.joblib")
    return tfidf


def loadTFIDF() -> TfidfVectorizer:
    return load(filename="../models/TfidfVectorizer.joblib")


def transformDataCV(cv: CountVectorizer, documents: Series) -> ndarray:
    print("Tranforming data...")
    data: ndarray = cv.transform(raw_documents=documents)
    print("Tranformed data ✅")
    return data


def transformDataTFIDF(tfidf: TfidfVectorizer, documents: Series) -> spmatrix:
    print("Tranforming data...")
    data: spmatrix = tfidf.transform(raw_documents=documents)
    print("Tranformed data ✅")
    return data


def trainModelCV(x: Series, y: Series) -> MultinomialNB:
    cv: CountVectorizer = createCV(documents=x)
    # cv: CountVectorizer = loadCV()

    x: ndarray = transformDataCV(cv=cv, documents=x)

    print("Training Multinomial Naive Bayes Model...")
    mnnb: MultinomialNB = MultinomialNB(force_alpha=True)
    mnnb.fit(X=x, y=y)
    print("Trained Multinomial Naive Bayes Model ✅")
    dump(value=mnnb, filename="../models/mnnbCV.joblib")
    return mnnb


def loadModelCV() -> MultinomialNB:
    return load(filename="../models/mnnbCV.joblib")


def trainModelTFIDF(x: Series, y: Series) -> MultinomialNB:
    tfidf: TfidfVectorizer = createTFIDF(documents=x)
    # tfidf: TfidfVectorizer = loadTFIDF()

    x: spmatrix = transformDataTFIDF(tfidf=tfidf, documents=x)

    print("Training Multinomial Naive Bayes Model...")
    mnnb: MultinomialNB = MultinomialNB(force_alpha=True)
    mnnb.fit(X=x, y=y)
    print("Trained Multinomial Naive Bayes Model ✅")
    dump(value=mnnb, filename="../models/mnnbTFIDF.joblib")
    return mnnb


def loadModelTFIDF() -> MultinomialNB:
    return load(filename="../models/mnnbTFIDF.joblib")


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

    mnnbCV: MultinomialNB = trainModelCV(x=xTrain, y=yTrain)
    mnnbTFIDF: MultinomialNB = trainModelTFIDF(x=xTrain, y=yTrain)


if __name__ == "__main__":
    main()
