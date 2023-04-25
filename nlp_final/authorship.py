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
from keras.utils import pad_sequences
from numpy import ndarray
from pandas import DataFrame, Series

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


def buildModel(
    inputDimensionSize: int, outputDimensionSize: int, inputLength: int
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
        loss=losses.categorical_crossentropy,
        metrics=["accuracy"],
    )
    model.compile()
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
    args: Namespace = getArgs()

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
    padSequences(
        tokenizer=tokenizer, text=df["text"], sequenceMaxLength=args.max_length
    )

    veaa: Sequential = buildModel(
        inputDimensionSize=args.input_dimension_size,
        outputDimensionSize=args.output_dimension_size,
        inputLength=args.max_length,
    )

    print("Training VEAA model...")
    veaa.fit(
        x=df["text"],
        y=df["author"],
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
