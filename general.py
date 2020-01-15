import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#from sklearn import


def get_corpus(df):
    corpus = []
    for i in range(df.shape[0]):
        corpus.append(df.iloc[i][3])
    return corpus

def equalize_array(a, b):
    """randomly reduces the larger array to the length of the smaller array"""
    size_a = a.size
    size_b = b.size

    big_arr = None
    lil_arr = None
    target_size = 0

    if size_a == size_b:
        return (a, b)
    elif size_a > size_b:
        big_arr = np.copy(a)
        lil_arr = np.copy(b)
        target_size = size_b
    else:
        big_arr = np.copy(b)
        lil_arr = np.copy(a)
        target_size = size_a

    big_arr = np.random.choice(big_arr, target_size, replace=False)

    return (lil_arr, big_arr)

