from argparse import Namespace
from pathlib import Path
from typing import Any, List

from gensim.models import FastText, Word2Vec
from joblib import dump, load
from numpy import ndarray
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, make_pipeline


def main(args: Namespace, x: DataFrame, y: DataFrame) -> None:
    vectorizer: TfidfVectorizer | Word2Vec | FastText = loadVectorizer(
        vectorizer=args.classifierVectorizer,
        vectorizerType=args.classifierVectorizerType,
    )

    pass


def loadVectorizer(
    vectorizer: Path, vectorizerType: str
) -> TfidfVectorizer | Word2Vec | FastText:
    match vectorizerType:
        case "tf-idf":
            model: TfidfVectorizer = load(filename=vectorizer)
        case "word2vec":
            model: Word2Vec = Word2Vec.load(vectorizer)
        case "fasttext":
            model: FastText = FastText.load(vectorizer)
    return model


def trainMultinomialNaiveBayes(x: ndarray, y: ndarray, outputPath: Path) -> None:
    parameters: dict[str, List[Any]] = {
        "multinomialnb__alpha": [1.0, 0.5, 1.5],
        "multinomialnb__force_alpha": [True],
    }

    pipeline: Pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())
    gscv: GridSearchCV = GridSearchCV(
        estimator=pipeline, param_grid=parameters, n_jobs=1
    )
    gscv.fit(X=x, y=y)
    model: Pipeline = gscv.best_estimator_
    dump(value=model, filename=Path(outputPath, "multinomialNB.joblib"))
