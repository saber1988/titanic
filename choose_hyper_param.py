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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score, train_test_split,StratifiedShuffleSplit
from sklearn.externals import joblib
from util import *

data_train = pd.read_csv("data/train.csv", dtype={"Age": np.float64})

# data_train, data_test = train_test_split(data_train, test_size=0.2)

# Fare
data_train.Fare = data_train.Fare.map(lambda x: np.nan if x == 0 else x)
data_train.loc[(data_train.Fare.isnull())&(data_train.Pclass == 1), 'Fare'] = \
    np.median(data_train[data_train['Pclass'] == 1]['Fare'].dropna())
data_train.loc[(data_train.Fare.isnull())&(data_train.Pclass == 2), 'Fare'] = \
    np.median(data_train[data_train['Pclass'] == 2]['Fare'].dropna())
data_train.loc[(data_train.Fare.isnull())&(data_train.Pclass == 3), 'Fare'] = \
    np.median(data_train[data_train['Pclass'] == 3]['Fare'].dropna())

# creating a title column from name
title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
            'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
            'Don', 'Jonkheer']
data_train['Title']=data_train['Name'].map(lambda x: substrings_in_string(x, title_list))


# Title
data_train['Title'] = data_train.apply(replace_titles, axis=1)
dummies_Title = pd.get_dummies(data_train['Title'], prefix='Title')
data_train = pd.concat([data_train, dummies_Title], axis=1)

# AgeFill
data_train['AgeFill'] = data_train['Age']
mean_ages = np.zeros(4)
mean_ages[0] = np.average(data_train[data_train['Title'] == 'Miss']['Age'].dropna())
mean_ages[1] = np.average(data_train[data_train['Title'] == 'Mrs']['Age'].dropna())
mean_ages[2] = np.average(data_train[data_train['Title'] == 'Mr']['Age'].dropna())
mean_ages[3] = np.average(data_train[data_train['Title'] == 'Master']['Age'].dropna())
data_train.loc[(data_train.Age.isnull()) & (data_train.Title == 'Miss'), 'AgeFill'] = mean_ages[0]
data_train.loc[(data_train.Age.isnull()) & (data_train.Title == 'Mrs'), 'AgeFill'] = mean_ages[1]
data_train.loc[(data_train.Age.isnull()) & (data_train.Title == 'Mr'), 'AgeFill'] = mean_ages[2]
data_train.loc[(data_train.Age.isnull()) & (data_train.Title == 'Master'), 'AgeFill'] = mean_ages[3]

# AgeCat
data_train.loc[(data_train.AgeFill <= 10), 'AgeCat'] = 'child'
data_train.loc[(data_train.AgeFill > 60), 'AgeCat'] = 'aged'
data_train.loc[(data_train.AgeFill > 10) & (data_train.AgeFill <= 30), 'AgeCat'] = 'adult'
data_train.loc[(data_train.AgeFill > 30) & (data_train.AgeFill <= 60), 'AgeCat'] = 'senior'
dummies_AgeCat = pd.get_dummies(data_train['AgeCat'], prefix='AgeCat')
data_train = pd.concat([data_train, dummies_AgeCat], axis=1)

# HasCabin
data_train["HasCabin"] = 1
data_train["HasCabin"][data_train.Cabin.isnull()] = 0

# Sex
le = preprocessing.LabelEncoder()
le.fit(data_train['Sex'] )
x_sex = le.transform(data_train['Sex'])
data_train['Sex'] = x_sex.astype(np.float)

# Pclass
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')
data_train = pd.concat([data_train, dummies_Pclass], axis=1)

# Embarked
data_train.Embarked = data_train.Embarked.fillna('S')
dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')
data_train = pd.concat([data_train, dummies_Embarked], axis=1)

train_data = data_train.loc[:, ['Sex', 'SibSp', 'Parch', 'Fare', 'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs',
                                'AgeCat_adult', 'AgeCat_aged', 'AgeCat_child', 'AgeCat_senior', 'HasCabin',
                                'Pclass_1', 'Pclass_2', 'Pclass_3', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]

train_data = preprocessing.scale(train_data)

train_label = data_train.loc[:, ['Survived']]

seed = 0
rf_clf = RandomForestClassifier(n_estimators=500, criterion='entropy', max_depth=5, min_samples_split=1,
                             min_samples_leaf=1, max_features='auto', bootstrap=False, oob_score=False, n_jobs=1,
                             random_state=seed, verbose=0)
pipeline = Pipeline([('clf', rf_clf)])
param_grid = dict(clf__n_estimators=[100, 500], clf__bootstrap=[False, True])
cv = StratifiedShuffleSplit(train_label, n_iter=10, test_size=0.2, train_size=None, random_state=seed)

grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=3,scoring='accuracy', cv=cv)
grid_search.fit(train_data, train_label.values.ravel())
print grid_search.best_estimator_
print grid_search.best_score_
print grid_search.best_estimator_.named_steps['clf'].feature_importances_
rf_clf = grid_search.best_estimator_

title = 'learning curve for random forest'
plot_learning_curve(grid_search.best_estimator_, title, train_data, train_label.values.ravel(), ylim=(0.7, 1.01), cv=cv, n_jobs=1)
# joblib.dump(grid_search.best_estimator_.named_steps['clf'], 'model/rf_model.pkl', compress=3)

svc_clf = SVC(kernel='rbf')

pipeline=Pipeline([('clf', svc_clf)])

param_grid = dict(clf__C=[10, 5])

grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=3,scoring='accuracy',
                           cv=StratifiedShuffleSplit(train_label, n_iter=10, test_size=0.2, train_size=None,
                                                     random_state=seed))
grid_search.fit(train_data, train_label.values.ravel())
print grid_search.best_estimator_
print grid_search.best_score_
svc_clf = grid_search.best_estimator_

lr_clf = LogisticRegression()

pipeline=Pipeline([('clf', lr_clf)])

param_grid = dict(clf__penalty=['l1', 'l2'], clf__C=[10, 5])

grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=3,scoring='accuracy',
                           cv=StratifiedShuffleSplit(train_label, n_iter=10, test_size=0.2, train_size=None,
                                                     random_state=seed))
grid_search.fit(train_data, train_label.values.ravel())
print grid_search.best_estimator_
print grid_search.best_score_
lr_clf = grid_search.best_estimator_

eclf = VotingClassifier(estimators=[('lr', lr_clf), ('rf', rf_clf), ('svc', svc_clf)], voting='hard')
scores = cross_val_score(eclf, train_data, train_label.values.ravel(), cv=5, scoring='accuracy')
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

joblib.dump(eclf, 'model/model.pkl', compress=3)