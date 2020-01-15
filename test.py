#import pandas as pd
#import numpy as np
#from matplotlib import pyplot as plt
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import train_test_split
##from sklearn import
#from general import *
import re

my_str = "This is #urboi speaking. I'm in cancun $ipping on a CO-CO-NUT!!!"
arr = my_str.split()
print(arr)
res = len(re.findall(r'\w+', my_str))
print('split %i' % len(arr))
print('re %i' % res)
