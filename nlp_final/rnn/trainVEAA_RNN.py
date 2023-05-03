from datetime import datetime
from pathlib import Path

import numpy
import pandas
from joblib import dump, load
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import LSTM, Dense, Embedding
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import sequence

TRAINING_DATA: Path = Path("../../dataset/trainingData_5-2-2023.csv")
TESTING_DATA: Path = Path("../../dataset/testingData_5-2-2023.csv")
LOG_FOLDER: Path = Path(
    "../logs/veaa-"
    + datetime.now().strftime(
        "%Y%m%d-%H%M%S",
    )
)

numpy.random.seed(42)


def createTokenizer(documents: Series) -> Tokenizer:
    print("Creating Tokenizer...")
    tokenizer: Tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts=documents)
    dump(value=tokenizer, filename="../../models/Tokenizer_RNN.joblib")
    print("Created Tokenizer ✅")
    return tokenizer


def loadTokenizer() -> Tokenizer:
    return load(filename="../../models/Tokenizer_RNN.joblib")


def transformData(tokenizer: Tokenizer, documents: Series) -> list[list]:
    print("Transforming data...")
    seq: list[list] = tokenizer.texts_to_sequences(texts=documents)
    seq = sequence.pad_sequences(seq, maxlen=1000)
    print("Transformed data ✅")
    return seq


def transformLabels(labels: Series) -> ndarray:
    print("Transforming labels...")
    labels: ndarray = to_categorical(y=labels, num_classes=50)
    print("Transformed labels ✅")
    return labels


def buildModel() -> Sequential:
    model = Sequential()
    model.add(
        Embedding(
            input_dim=10001,
            output_dim=32,
            input_length=1000,
        )
    )
    model.add(LSTM(units=100))
    model.add(Dense(units=50, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model


def trainModel(
    xTrain: list[list],
    yTrain: ndarray,
    xTest: list[list],
    yTest: ndarray,
    model: Sequential,
) -> None:
    mc: ModelCheckpoint = ModelCheckpoint(
        filepath="../../models/veaaRNN.h5",
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        mode="max",
    )

    tb: TensorBoard = TensorBoard(
        log_dir=LOG_FOLDER,
        histogram_freq=1,
        write_images=True,
    )

    print("Training model...")

    model.fit(
        x=xTrain,
        y=yTrain,
        batch_size=32,
        epochs=30,
        callbacks=[mc, tb],
        validation_data=(xTest, yTest),
    )

    print("Trained model ✅")


def main() -> None:
    trainingDF: DataFrame = pandas.read_csv(filepath_or_buffer=TRAINING_DATA)

    xTrain, xTest, yTrain, yTest = train_test_split(
        trainingDF["text"],
        trainingDF["author"],
        test_size=0.15,
        train_size=0.85,
        random_state=42,
        shuffle=True,
    )

    # tokenizer: Tokenizer = createTokenizer(documents=xTrain)
    tokenizer: Tokenizer = loadTokenizer()

    xTrain: list[list] = transformData(tokenizer=tokenizer, documents=xTrain)
    yTrain: ndarray = transformLabels(labels=yTrain)

    model: Sequential = buildModel()

    trainModel(
        xTrain=xTrain,
        yTrain=yTrain,
        xTest=xTest,
        yTest=yTest,
        model=model,
    )

    exit()


if __name__ == "__main__":
    main()

# data: DataFrame = pandas.read_csv(
#     "../../dataset/trainingData_5-2-2023.csv",
#     usecols=["text", "author"],
#     sep=",",
#     encoding_errors="ignore",
# )
# texts: Series = data["text"]
# labels: Series = data["author"] - 1

# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(texts)
# sequences = tokenizer.texts_to_sequences(texts)


# # Pad the sequences to a fixed length
# max_sequence_length = 1000  # Set the maximum sequence length
# sequences = sequence.pad_sequences(
#     sequences,
#     maxlen=max_sequence_length,
# )


# # Encode the labels into one-hot vectors
# num_classes = 50  # Set the number of classes
# labels = to_categorical(
#     labels,
#     num_classes,
# )

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(
#     sequences,
#     labels,
#     test_size=0.2,
#     random_state=42,
# )

# test_text = pandas.DataFrame(X_test)
# test_labels = pandas.DataFrame(y_test)

# test_text.to_csv("../dataset/test_text.csv")
# test_labels.to_csv("../dataset/test_labels.csv")


# # load the dataset but only keep the top n words, zero the rest
# top_words = 10001

# filepath = "../models/authorship_rnn_model_40_epoch.h5"
# checkpoint = ModelCheckpoint(
#     filepath,
#     monitor="val_loss",
#     verbose=1,
#     save_best_only=True,
#     mode="min",
# )

# # truncate and pad input sequences
# """
# max_review_length = 1000
# X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
# X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# """

# # create the model
# embedding_vecor_length = 32
# model = Sequential()
# model.add(
#     Embedding(
#         top_words,
#         embedding_vecor_length,
#         input_length=max_sequence_length,
#     )
# )
# model.add(LSTM(100))
# model.add(
#     Dense(
#         50,
#         activation="softmax",
#     )
# )  # Output layer with 50 units and softmax activation
# model.compile(
#     loss="categorical_crossentropy",
#     optimizer="adam",
#     metrics=["accuracy"],
# )  # Use categorical_crossentropy loss for multiclass classification
# print(model.summary())

# logFolder: Path = Path(
#     "../logs/veaa-"
#     + datetime.now().strftime(
#         "%Y%m%d-%H%M%S",
#     )
# )

# tensorboard_callback: TensorBoard = TensorBoard(
#     log_dir=logFolder,
#     histogram_freq=1,
#     write_images=True,
# )

# model.fit(
#     X_train,
#     y_train,
#     validation_data=(X_test, y_test),
#     epochs=30,
#     batch_size=32,
#     callbacks=[checkpoint, tensorboard_callback],
# )

# # Final evaluation of the model
# scores = model.evaluate(
#     X_test,
#     y_test,
#     verbose=0,
# )
# print("Accuracy: %.2f%%" % (scores[1] * 100))
