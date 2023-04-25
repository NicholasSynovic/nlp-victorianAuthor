import os
from argparse import ArgumentParser, Namespace
from logging import ERROR, Logger
from pathlib import Path
from warnings import filterwarnings

import numpy
import tensorflow
from dataHandler import loadData
from keras import losses
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, Dense, Embedding, TextVectorization
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, to_categorical
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
tfLogger: Logger = tensorflow.get_logger()
tfLogger.setLevel(level=ERROR)

filterwarnings(action="always")

TRAINING_DATASET: Path = Path(
    "../dataset/Gungor_2018_VictorianAuthorAttribution_data-train.csv"
)


def getArgs() -> Namespace:
    parser: ArgumentParser = ArgumentParser(prog="RNN Training for NLP Class")
    parser.add_argument(
        "-m",
        "--max-length",
        type=int,
        default=1000,
        required=False,
        help="Padding value to ensure that all text sequences are the same length",
    )
    parser.add_argument(
        "-i",
        "--input-dimension-size",
        type=int,
        default=1000,
        required=False,
        help="Input dimension size into the RNN's first Embedding layer",
    )
    parser.add_argument(
        "-o",
        "--output-dimension-size",
        type=int,
        default=32,
        required=False,
        help="Output dimension size from the RNN's first Embedding layer",
    )
    return parser.parse_args()


def createTokenizer(text: Series) -> Tokenizer:
    tokenizer: Tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts=text)

    return tokenizer


def buildModel(
    inputDimensionSize: int, outputDimensionSize: int, inputLength: int
) -> None:
    model: Sequential = Sequential()
    model.add(
        layer=Embedding(
            input_dim=inputDimensionSize,
            output_dim=outputDimensionSize,
            input_length=inputLength,
        )
    )
    model.add(
        layer=LSTM(
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
        optimizer="adam",
        loss=losses.categorical_crossentropy,
        metrics=["accuracy"],
    )
    model.compile()
    model.summary()

    print(type(model))


def main() -> None:
    args: Namespace = getArgs()

    numpy.random.seed(seed=42)

    df: DataFrame = loadData(path=TRAINING_DATASET)

    tokenizer: Tokenizer = createTokenizer(text=df["text"])
    textSequences: list[list] = tokenizer.texts_to_sequences(texts=df["text"])

    paddedSequences: ndarray = pad_sequences(
        sequences=textSequences, maxlen=args["max_length"]
    )

    model = buildModel(
        inputDimensionSize=args["input_dimension_size"],
        outputDimensionSize=args["output_dimension_size"],
        inputLength=args["max_length"],
    )


if __name__ == "__main__":
    main()

quit()


# # load the dataset but only keep the top n words, zero the rest
# top_words = 10001

# filepath = "authorship_rnn_model.h5"
# checkpoint = ModelCheckpoint(
#     filepath, monitor="val_loss", verbose=1, save_best_only=True, mode="min"
# )

# truncate and pad input sequences
"""
max_review_length = 1000
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
"""

# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(
    Embedding(top_words, embedding_vecor_length, input_length=max_sequence_length)
)
model.add(LSTM(100))
model.add(
    Dense(50, activation="softmax")
)  # Output layer with 50 units and softmax activation
model.compile(
    loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)  # Use categorical_crossentropy loss for multiclass classification
print(model.summary())
model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=64,
    callbacks=[checkpoint],
)


# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))
