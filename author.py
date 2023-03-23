import pandas as pd
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

#Dataset available at https://archive.ics.uci.edu/ml/machine-learning-databases/00454/
trainData = pd.read_csv('DataSets/Gungor_2018_VictorianAuthorAttribution_data-train.csv',sep = ',',  encoding ='latin1')

'''
#use label encoder to assign numerical codes to the authors
LE = LabelEncoder()       
LE.fit( trainData['author'] )         
train_Y = LE.transform(trainData['author'])


#create map to associate the numerical code to each author
keyMap = {}
for i, j in zip(train_Y, trainData['author']):
    keyMap[i] = j

'''

#create tf-idf vector using training data
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(trainData['text'])

#Vector based on frequency, not needed for TF-IDF vector
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)

#Vector based on TF-IDF
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#train multinomial naive bayes classifier with training data
clf = MultinomialNB().fit(X_train_tfidf, trainData['author'])


#read book from csv file to return the author
testData = pd.read_csv('test-snippets.tsv', sep = '\t', names = ['author', 'passage'])
X_new_counts = count_vect.transform(testData['passage'])
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)


