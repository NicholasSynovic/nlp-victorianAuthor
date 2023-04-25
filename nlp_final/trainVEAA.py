from datetime import datetime
from pathlib import Path

import numpy
from dataHandler import loadData
from keras.callbacks import TensorBoard
from keras.layers import LSTM, Dense, Embedding
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, to_categorical
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split

numpy.random.seed(seed=42)
TRAINING_DATASET: Path = Path(
    "../dataset/Gungor_2018_VictorianAuthorAttribution_data-train.csv"
)


def createSequences(texts: Series) -> ndarray:
    print("Tokenizing texts...")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts=texts)
    sequences = tokenizer.texts_to_sequences(texts=texts)
    print("Tokenizied texts ✅")
    return pad_sequences(
        sequences=sequences,
        maxlen=1000,
    )


def createModel() -> Sequential:
    print("Building model...")
    model = Sequential()
    model.add(
        Embedding(
            input_dim=10001,
            output_dim=32,
            input_length=1000,
        )
    )
    model.add(
        LSTM(
            units=100,
            activation="relu",
        )
    )
    model.add(
        Dense(
            units=50,
            activation="softmax",
        )
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    model.summary()
    print("Built model ✅\n")
    return model


def main() -> None:
    pbPath: Path = Path("models/veaa").resolve()
    logFolder: Path = Path("../logs/veaa-" + datetime.now().strftime("%Y%m%d-%H%M%S"))

    tensorboardCallback: TensorBoard = TensorBoard(
        log_dir=logFolder,
        histogram_freq=1,
        write_images=True,
    )

    print()
    df: DataFrame = loadData(path=TRAINING_DATASET)
    texts: Series = df["text"]
    labels: Series = df["author"] - 1
    labels = to_categorical(labels, 50)

    sequences: ndarray = createSequences(texts=texts)

    xTrain, xTest, yTrain, yTest = train_test_split(
        sequences,
        labels,
        test_size=0.15,
        train_size=0.85,
        random_state=42,
    )

    veaa: Sequential = createModel()
    veaa.fit(
        x=xTrain,
        y=yTrain,
        batch_size=64,
        epochs=30,
        validation_data=(xTest, yTest),
        callbacks=[tensorboardCallback],
    )

    veaa.save(
        filepath=pbPath,
        overwrite=True,
        save_format="tf",
    )


if __name__ == "__main__":
    main()
