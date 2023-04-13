from gensim.models import FastText, Word2Vec
from numpy import ndarray
from sklearn.feature_extraction.text import TfidfVectorizer


def trainTFIDF(x: ndarray) -> TfidfVectorizer:
    tfidf: TfidfVectorizer = TfidfVectorizer(
        decode_error="ignore", lowercase=False, ngram_range=(1, 3), norm="l2"
    )

    print("Training TF-IDF vectorizer...")
    tfidf.fit(raw_documents=x)
    return tfidf


def trainWord2Vec(x: ndarray) -> Word2Vec:
    print("Training Word2Vec vectorizer...")
    w2v: Word2Vec = Word2Vec(sentences=x, vector_size=500)
    return w2v


def trainFastText(x: ndarray) -> FastText:
    print("Training FastText vectorizer...")
    ft: FastText = FastText(sentences=x, vector_size=500)
    return ft
