from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path

import numpy
from dataHandler import loadData
from keras import losses
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import LSTM, Dense, Embedding
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, to_categorical
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split

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
        default=10001,
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
    print("Tokenizing dataset...")
    tokenizer: Tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts=text)
    print("Tokenized dataset ✅")

    return tokenizer


def padSequences(tokenizer: Tokenizer, text: Series, sequenceMaxLength: int) -> ndarray:
    print("Padding sequences...")
    textSequences: list[list] = tokenizer.texts_to_sequences(texts=text)
    paddedSequences: ndarray = pad_sequences(
        sequences=textSequences, maxlen=sequenceMaxLength
    )
    print("Padded sequences ✅")
    return paddedSequences


def createSplits(sequences: ndarray, labels: Series) -> dict[list]:
    print("Splitting data...")

    xTrain, xTest, yTrain, yTest = train_test_split(
        sequences,
        labels,
        test_size=0.15,
        train_size=0.85,
        random_state=42,
        shuffle=True,
    )

    print("Split data ✅")
    return {
        "xTrain": xTrain,
        "yTrain": yTrain,
        "xTest": xTest,
        "yTest": yTest,
    }


def buildModel(
    inputDimensionSize: int = 10001,
    outputDimensionSize: int = 32,
    inputLength: int = 1000,
) -> Sequential:
    print("Creating model...")
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
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    print("Created model ✅")
    return model


def saveModel(model: Sequential) -> None:
    pbPath: Path = Path("models/veaa").resolve()

    print(f"Saving model to: {pbPath.__str__()}")

    model.save(
        filepath=pbPath,
        overwrite=True,
        save_format="tf",
    )

    print(f"Saved model to: {pbPath.__str__()} ✅")


def main() -> None:
    # args: Namespace = getArgs()

    numpy.random.seed(seed=42)

    logFolder: Path = Path("logs/veaa-" + datetime.now().strftime("%Y%m%d-%H%M%S"))

    tensorboardCallback: TensorBoard = TensorBoard(
        log_dir=logFolder,
        histogram_freq=1,
        write_images=True,
    )

    print("\n")
    df: DataFrame = loadData(path=TRAINING_DATASET)

    tokenizer: Tokenizer = createTokenizer(text=df["text"])
    sequences: ndarray = padSequences(
        tokenizer=tokenizer,
        text=df["text"],
        sequenceMaxLength=1000,
    )

    splits: dict[list] = createSplits(sequences=sequences, labels=df["author"])

    veaa: Sequential = buildModel()

    print("Training VEAA model...")
    veaa.fit(
        x=splits["xTrain"],
        y=splits["yTrain"],
        batch_size=1000,
        epochs=100,
        callbacks=[tensorboardCallback],
        validation_split=0.15,
    )
    print("Trained VEAA model ✅")

    saveModel(model=veaa)


if __name__ == "__main__":
    main()

# # Final evaluation of the model
# scores = model.evaluate(X_test, y_test, verbose=0)
# print("Accuracy: %.2f%%" % (scores[1] * 100))
