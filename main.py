import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
#from sklearn import

df = pd.read_csv('train.csv')

y_train = []

corpus = []
for i in range(df.shape[0]):
    corpus.append(df.iloc[i][3])
    y_train.append(df.iloc[i][4])
vectorizer = CountVectorizer()
X_vectex_train = vectorizer.fit_transform(corpus)

print('checkpoint')

log_reg = LogisticRegression(random_state=0)
log_reg.fit(X_vectex_train, y_train)
y_pred = log_reg.predict(X_vectex_train)
print(log_reg.score(y_train, y_pred))



