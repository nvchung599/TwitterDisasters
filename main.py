import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#from sklearn import

df = pd.read_csv('train.csv')

mask = np.random.rand(len(df)) < 0.7
df_train = df[mask]
df_test = df[~mask]

"""TRAINTRAINTRAINTRAINTRAINTRAINTRAINTRAINTRAINTRAINTRAINTRAINTRAINTRAIN"""
corpus = []
for i in range(df_train.shape[0]):
    corpus.append(df.iloc[i][3])

vectorizer = CountVectorizer()
x_train = vectorizer.fit_transform(corpus)
y_train = df_train.to_numpy()[:,4].astype('int')


log_reg = LogisticRegression(random_state=0, max_iter=1000)
log_reg.fit(x_train, y_train)
print('\nTRAIN ACCURACY')
print(log_reg.score(x_train, y_train))
#y_pred = log_reg.predict(X_vectex_train)


"""TESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTESTTEST"""
corpus = []
for i in range(df_test.shape[0]):
    corpus.append(df.iloc[i][3])

x_test = vectorizer.transform(corpus)
y_test = df_test.to_numpy()[:,4].astype('int')

print('\nTEST ACCURACY')
print(log_reg.score(x_test, y_test))




