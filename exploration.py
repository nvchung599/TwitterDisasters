import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#from sklearn import

df = pd.read_csv('train.csv')

print(df.columns)

print(df['target'].value_counts())

# label distribution
# character length distribution
# word qty distribution
# word length distribution
# qty stopwords
# punctuation bar graph
# word types??????? possessive, nouns, etc
# unique word distribution/bar graph
# capitalization distribution??????
# Ngram analysis

# cleanup:
#   spelling
#   remove urls
#   remove/detect emojis
#   spelling
#   spelling

# GloVe vectorization, see link
# https://medium.com/analytics-vidhya/basics-of-using-pre-trained-glove-vectors-in-python-d38905f356db
