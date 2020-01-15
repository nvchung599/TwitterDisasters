TOGGLE = 1

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#from sklearn import
from general import *


df = pd.read_csv('train.csv')

if TOGGLE == 1:

    print('\nDF COLUMNS')
    print(df.columns)

    print('\nLABEL DISTRIBUTION')
    print(df['target'].value_counts())

    print('\nCHARACTER LENGTH DISTRIBUTION')
    print('see plot')

    y = df.iloc[:,4].astype('int')
    y_0 = y[y==0]
    y_1 = y[y==1]

    corpus = get_corpus(df)
    char_lengths = []
    for tweet in corpus:
        char_lengths.append(len(tweet))
    x = np.array(char_lengths)
    x_0 = x[y==0]
    x_1 = x[y==1]

    bins = np.linspace(0,200,100)

    plt.rcParams["patch.force_edgecolor"] = True
    plt.hist(x_0, weights=np.ones(len(x_0))/len(x_0), bins=bins, alpha=0.5, label='0')
    plt.hist(x_1, weights=np.ones(len(x_1))/len(x_1), bins=bins, alpha=0.5, label='1')
    plt.title('Character Length Distribution of Disasters (1) vs non-Disasters (0)')
    plt.xlabel('Character Length')
    plt.ylabel('% of tweets')
    plt.legend()
    #plt.scatter(np.array(char_lengths), df.iloc[:,4].astype('int'))
    plt.show()

if TOGGLE == 2:
    print('bobobobobobobobobobobobob')

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
