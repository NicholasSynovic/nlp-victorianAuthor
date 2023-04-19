from pathlib import Path

trainingDataPath: Path = Path("../dataset/training.csv")

tfidfPath: Path = Path("../models/tfidf.joblib").resolve()
word2vecPath: Path = Path("../models/word2vec.gensim").resolve()
fasttextPath: Path = Path("../models/fasttext.gensim").resolve()
