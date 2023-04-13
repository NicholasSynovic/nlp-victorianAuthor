from argparse import Namespace
from pathlib import Path
from typing import Any, List

from joblib import dump
from numpy import ndarray
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearnex import patch_sklearn

patch_sklearn()


def main(
    args: Namespace,
    x: ndarray,
    y: ndarray,
    validationX: ndarray,
    validationY: ndarray,
) -> None:
    if args.vectorizerTrainTFIDF:
        trainTFIDF(x=x, y=y, outputPath=args.vectorizerOutput)


def trainTFIDF(x: ndarray, y: ndarray, outputPath: Path) -> None:
    parameters: dict[str, List[Any]] = {
        "tfidfvectorizer__decode_error": ["ignore"],
        "tfidfvectorizer__lowercase": [False, True],
        "tfidfvectorizer__ngram_range": [(1, 3)],
        "tfidfvectorizer__norm": ["l1", "l2"],
    }

    gscv: GridSearchCV = GridSearchCV(
        estimator=TfidfVectorizer(), param_grid=parameters, n_jobs=1
    )

    print("Training TF-IDF vectorizer...")
    gscv.fit(X=x, y=y)
    model: TfidfVectorizer = gscv.best_estimator_
    dump(value=model, filename=Path(outputPath, "tfidf.joblib"))
