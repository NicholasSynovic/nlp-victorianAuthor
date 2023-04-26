from datetime import datetime
from pathlib import Path

import pandas
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

# fix random seed for reproducibility
tf.random.set_seed(7)

# Load the data
data = pandas.read_csv("training.csv", usecols=["text", "author"], sep=",")
texts = data["text"]  # Extract the text column
labels = data["author"] - 1  # Extract the label column

# Convert the text into numerical sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)


# Pad the sequences to a fixed length
max_sequence_length = 1000  # Set the maximum sequence length
sequences = sequence.pad_sequences(sequences, maxlen=max_sequence_length)


# Encode the labels into one-hot vectors
num_classes = 50  # Set the number of classes
labels = to_categorical(labels, num_classes)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    sequences, labels, test_size=0.2, random_state=42
)

test_text = pandas.DataFrame(X_test)
test_labels = pandas.DataFrame(y_test)

test_text.to_csv("test_text.csv")
test_labels.to_csv("test_labels.csv")


# load the dataset but only keep the top n words, zero the rest
top_words = 10001

filepath = "authorship_rnn_model_40_epoch.h5"
checkpoint = ModelCheckpoint(
    filepath, monitor="val_loss", verbose=1, save_best_only=True, mode="min"
)

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

logFolder: Path = Path("logs/lenet-" + datetime.now().strftime("%Y%m%d-%H%M%S"))

tensorboard_callback: TensorBoard = TensorBoard(
    log_dir=logFolder,
    histogram_freq=1,
    write_images=True,
)

model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=32,
    callbacks=[checkpoint, tensorboard_callback],
)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))
