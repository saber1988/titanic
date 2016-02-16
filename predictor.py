# -*- coding: utf-8 -*-
__author__ = 'shidaiting01'

import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
from sklearn.externals import joblib
from util import *

data_test = pd.read_csv("data/test.csv")
ID = data_test['PassengerId']

clf = joblib.load('model/model.pkl')
data = preprocess_data(data_test)
label = clf.predict(data).astype(int)
df = DataFrame(dict(Survived=label, PassengerId=ID), columns=['PassengerId', 'Survived'])
df.to_csv("data/prediction.csv", index=False)