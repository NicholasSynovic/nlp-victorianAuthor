import pandas
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence

# load the model from file
model = load_model("../models/veaaCNN.h5")

# example passage to classify, using line from old man and the sea
new_passage = [
    "He no longer dreamed of storms, nor of women, nor of great occurrences, nor of great fish, nor fights, nor contests of strength, nor of his wife. He only dreamed of places now and of the lions on the beach. They played like young cats in the dusk and he loved them as he loved the boy."
]

# preprocess the passage
top_words = 100001
max_review_length = 1000
tokenizer = Tokenizer(num_words=top_words)
tokenizer.fit_on_texts(new_passage)
Sequence = tokenizer.texts_to_sequences(new_passage)
padded_sequence = sequence.pad_sequences(Sequence, maxlen=max_review_length)

# make prediction
author_classification = pandas.DataFrame(
    enumerate(model.predict(padded_sequence)[0]), columns=["Author", "Score (%)"]
)

# retrieve author, label mapping file
label_mapping = pandas.read_excel(
    "Author_Label_Mapping.xlsx", header=None, names=["key", "value"]
)
dictionary = label_mapping.set_index("key")["value"].to_dict()

# Format output
author_classification = author_classification.sort_values(
    by="Score (%)", ascending=False
)
author_classification["Author"] = (author_classification["Author"] + 1).map(dictionary)
author_classification["Score (%)"] = author_classification["Score (%)"] * 100

print(author_classification.head(5))

print("Correct Author: Ernest Hemingway")
