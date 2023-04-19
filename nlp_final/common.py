from pathlib import Path
from typing import Tuple

from gensim.models import FastText, Word2Vec
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer

trainingDataPath: Path = Path("../dataset/training.csv")

tfidfPath: Path = Path("../models/tfidf.joblib").resolve()
word2vecPath: Path = Path("../models/word2vec.gensim").resolve()
fasttextPath: Path = Path("../models/fasttext.gensim").resolve()


def loadVectorizers() -> Tuple[FastText, TfidfVectorizer, Word2Vec]:
    fasttext: FastText = FastText.load(fasttextPath.__str__())
    tfidf: TfidfVectorizer = load(filename=tfidfPath)
    word2vec: Word2Vec = Word2Vec.load(word2vecPath.__str__())
    return (fasttext, tfidf, word2vec)
