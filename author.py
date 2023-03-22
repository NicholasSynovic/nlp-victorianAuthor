import pandas as pd
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

trainData = pd.read_csv('DataSets/Gungor_2018_VictorianAuthorAttribution_data-train.csv',sep = ',',  encoding ='latin1')
print(trainData)
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

tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#train multinomial naive bayes classifier with training data
clf = MultinomialNB().fit(X_train_tfidf, trainData['author'])


#read book from csv file to return the author
testData = pd.read_csv('test-snippets.tsv', sep = '\t', names = ['author', 'passage'])
X_new_counts = count_vect.transform(testData['passage'])
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)


