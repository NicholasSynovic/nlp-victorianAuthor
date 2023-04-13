from argparse import Namespace
from pathlib import Path
from typing import Any, List

from gensim.models import Doc2Vec, FastText, Word2Vec
from joblib import dump
from numpy import ndarray
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV


def main(
    args: Namespace,
    x: ndarray,
    y: ndarray,
) -> None:
    if args.vectorizerTrainTFIDF:
        trainTFIDF(
            x=x, y=y, numberOfJobs=args.vectorizerJobs, outputPath=args.vectorizerOutput
        )

    if args.vectorizerTrainWord2Vec:
        trainWord2Vec(x=x, outputPath=args.vectorizerOutput)

    if args.vectorizerTrainDoc2Vec:
        trainWord2Vec(x=x, outputPath=args.vectorizerOutput)

    if args.vectorizerTrainDoc2Vec:
        trainFastText(x=x, outputPath=args.vectorizerOutput)


def trainTFIDF(x: ndarray, y: ndarray, numberOfJobs: int, outputPath: Path) -> None:
    parameters: dict[str, List[Any]] = {
        "tfidfvectorizer__decode_error": ["ignore"],
        "tfidfvectorizer__lowercase": [False, True],
        "tfidfvectorizer__ngram_range": [(1, 3)],
        "tfidfvectorizer__norm": ["l1", "l2"],
    }

    gscv: GridSearchCV = GridSearchCV(
        estimator=TfidfVectorizer(), param_grid=parameters, n_jobs=numberOfJobs
    )

    print("Training TF-IDF vectorizer...")
    gscv.fit(X=x, y=y)
    model: TfidfVectorizer = gscv.best_estimator_
    dump(value=model, filename=Path(outputPath, "tfidf.joblib"))


def trainWord2Vec(x: ndarray, outputPath: Path) -> None:
    print("Training Word2Vec vectorizer...")
    w2v: Word2Vec = Word2Vec(sentences=x, vector_size=500)
    w2v.save(Path(outputPath, "word2vec.gensim").absolute().__str__())


def trainDoc2Vec(x: ndarray, outputPath: Path) -> None:
    print("Training Doc2Vec vectorizer...")
    d2v: Doc2Vec = Doc2Vec(documents=x, vector_size=500)
    d2v.save(Path(outputPath, "doc2vec.gensim").absolute().__str__())


def trainFastText(x: ndarray, outputPath: Path) -> None:
    print("Training FastText vectorizer...")
    ft: FastText = FastText(documents=x, vector_size=500)
    ft.save(Path(outputPath, "fasttext.gensim").absolute().__str__())
