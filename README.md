Victorian Era Author Recommender

 - Classifier that accepts a novel or passage and returns a suggested author from the Victorian era
 - Utilizes Scikit-Learn library for text classification and pandas library for data formatting


Dataset Citation:
GUNGOR, ABDULMECIT, Benchmarking Authorship Attribution Techniques Using Over A Thousand Books by Fifty Victorian Era Novelists, Purdue Master of Thesis, 2018-04
https://archive.ics.uci.edu/ml/datasets/Victorian+Era+Authorship+Attribution

## Scope

## Project Goal

Enter text and recommends an author back to you based off of the raw text inputted.

Training Steps

1. Create word embeddngs of the dataset
2. Use the embeddings to classify text to an author

Implentation Step

1. Read in raw text
2. convert to word embeding
3. Classify embedding
