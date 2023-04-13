from argparse import Namespace
from pathlib import Path

from gensim.models import FastText, Word2Vec
from joblib import dump
from numpy import ndarray
from sklearn.feature_extraction.text import TfidfVectorizer


def main(
    args: Namespace,
    x: ndarray,
) -> None:
    if args.vectorizerTrainTFIDF:
        trainTFIDF(x=x, outputPath=args.vectorizerOutput)

    if args.vectorizerTrainWord2Vec:
        trainWord2Vec(x=x, outputPath=args.vectorizerOutput)

    if args.vectorizerTrainFastText:
        trainFastText(x=x, outputPath=args.vectorizerOutput)


def trainTFIDF(x: ndarray, outputPath: Path) -> None:
    tfidf: TfidfVectorizer = TfidfVectorizer(
        decode_error="ignore", lowercase=False, ngram_range=(1, 3), norm="l2"
    )

    print("Training TF-IDF vectorizer...")
    tfidf.fit(raw_documents=x)
    dump(value=tfidf, filename=Path(outputPath, "tfidf.joblib"))


def trainWord2Vec(x: ndarray, outputPath: Path) -> None:
    print("Training Word2Vec vectorizer...")
    w2v: Word2Vec = Word2Vec(sentences=x, vector_size=500)
    w2v.save(Path(outputPath, "word2vec.gensim").absolute().__str__())


def trainFastText(x: ndarray, outputPath: Path) -> None:
    print("Training FastText vectorizer...")
    ft: FastText = FastText(sentences=x, vector_size=500)
    ft.save(Path(outputPath, "fasttext.gensim").absolute().__str__())
