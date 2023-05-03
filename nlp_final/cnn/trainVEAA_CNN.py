from datetime import datetime
from multiprocessing import cpu_count
from pathlib import Path

import numpy
import pandas
from joblib import dump, load
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Conv1D, Dense, Dropout, Embedding, GlobalMaxPooling1D
from keras.metrics import Precision, Recall
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import sequence

TRAINING_DATA: Path = Path("../../dataset/training.csv")
TESTING_DATA: Path = Path("../../dataset/testing.csv")
LOG_FOLDER: Path = Path(
    "../../logs/veaa-"
    + datetime.now().strftime(
        "%Y%m%d-%H%M%S",
    )
)

numpy.random.seed(42)


def createTokenizer(documents: Series) -> Tokenizer:
    print("Creating Tokenizer...")
    tokenizer: Tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts=documents)
    dump(value=tokenizer, filename="../../models/Tokenizer_CNN.joblib")
    print("Created Tokenizer ✅")
    return tokenizer


def loadTokenizer() -> Tokenizer:
    return load(filename="../../models/Tokenizer_CNN.joblib")


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
    rmsProp: RMSprop = RMSprop(lr=0.001)

    model = Sequential()
    model.add(
        Embedding(
            input_dim=10001,
            output_dim=128,
            input_length=1000,
        )
    )
    model.add(
        Conv1D(
            filters=256,
            kernel_size=3,
            activation="relu",
        )
    )
    model.add(GlobalMaxPooling1D())
    model.add(Dense(units=512, activation="relu"))
    model.add(Dropout(rate=0.25))
    model.add(Dense(units=50, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy",
        optimizer=rmsProp,
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
        filepath="../../models/veaaCNN.h5",
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
        epochs=5,
        callbacks=[mc, tb],
        validation_data=(xTest, yTest),
        # workers=cpu_count() // 2,
    )

    print("Trained model ✅")


def loadModel() -> Sequential:
    return load_model(filepath="../../models/veaaCNN.h5")


def evaluateModel(x: list[list], y: ndarray, crY: Series, model: Sequential) -> None:
    prediction: ndarray = model.predict(x=x, workers=cpu_count() // 2)

    accuracy: float = model.get_metrics_result()["accuracy"]

    precision: Precision = Precision()
    recall: Recall = Recall()

    precision.update_state(y_true=y, y_pred=prediction)
    recall.update_state(y_true=y, y_pred=prediction)

    precision: float = float(precision.result())
    recall: float = float(recall.result())

    f1Score: float = 2 * ((precision * recall) / (precision + recall))

    prediction: ndarray = numpy.argmax(a=prediction, axis=1)
    matrix: ndarray = confusion_matrix(y_true=crY, y_pred=prediction)
    perClassAccuracy: Series = Series(matrix.diagonal() / matrix.sum(axis=1))
    perClassAccuracy.index = numpy.arange(1, len(perClassAccuracy) + 1)
    perClassAccuracy.sort_values(inplace=True)

    print("Least accurate classes: \n", perClassAccuracy.head(n=5))
    print("Most accurate classes: \n", perClassAccuracy.tail(n=5))

    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-Score: ", f1Score)


def main() -> None:
    trainingDF: DataFrame = pandas.read_csv(filepath_or_buffer=TRAINING_DATA)
    testingDF: DataFrame = pandas.read_csv(filepath_or_buffer=TESTING_DATA)

    xTrain, xValidation, yTrain, yValidation = train_test_split(
        trainingDF["text"],
        trainingDF["author"],
        test_size=0.15,
        train_size=0.85,
        random_state=42,
        shuffle=True,
    )

    tokenizer: Tokenizer = loadTokenizer()

    # Uncomment out the below section to train a tokenizer from scratch
    # tokenizer: Tokenizer = createTokenizer(documents=xTrain)

    xTrain: list[list] = transformData(tokenizer=tokenizer, documents=xTrain)
    xValidation: list[list] = transformData(tokenizer=tokenizer, documents=xValidation)
    xTest: list[list] = transformData(tokenizer=tokenizer, documents=testingDF["text"])
    yTrain: ndarray = transformLabels(labels=yTrain - 1)
    yValidation: ndarray = transformLabels(labels=yValidation - 1)
    yTest: ndarray = transformLabels(labels=testingDF["author"] - 1)

    model: Sequential = loadModel()

    # Uncomment out the below section to train a model from scratch
    # model: Sequential = buildModel()

    # trainModel(
    #     xTrain=xTrain,
    #     yTrain=yTrain,
    #     xTest=xValidation,
    #     yTest=yValidation,
    #     model=model,
    # )

    evaluateModel(x=xTest, y=yTest, crY=testingDF["author"] - 1, model=model)


if __name__ == "__main__":
    main()
