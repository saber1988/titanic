# -*- coding: utf-8 -*-
__author__ = 'shidaiting01'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.learning_curve import learning_curve
from sklearn import preprocessing

def substrings_in_string(big_string, substrings):
    import string
    for substring in substrings:
        if string.find(big_string, substring) != -1:
            return substring
    print big_string
    return np.nan

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.
    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
    An object of that type which is cloned for each validation.
    title : string
    Title for the chart.
    X : array-like, shape (n_samples, n_features)
    Training vector, where n_samples is the number of samples and
    n_features is the number of features.
    y : array-like, shape (n_samples) or (n_samples, n_features), optional
    Target relative to X for classification or regression;
    None for unsupervised learning.
    ylim : tuple, shape (ymin, ymax), optional
    Defines minimum and maximum yvalues plotted.
    cv : integer, cross-validation generator, optional
    If an integer is passed, it is the number of folds (defaults to 3).
    Specific cross-validation objects can be passed, see
    sklearn.cross_validation module for the list of possible objects
    n_jobs : integer, optional
    Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    plt.savefig("plot/learning_curve.png")


# replacing all titles with mr, mrs, miss, master
def replace_titles(x):
    title=x['Title']
    if title in ['Mr','Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return 'Mr'
    elif title in ['Master']:
        return 'Master'
    elif title in ['Countess', 'Mme','Mrs']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms','Miss']:
        return 'Miss'
    elif title =='Dr':
        if x['Sex']=='Male':
            return 'Mr'
        else:
            return 'Mrs'
    elif title == '':
        if x['Sex']=='Male':
            return 'Master'
        else:
            return 'Miss'
    else:
        return title


def preprocess_data(df):
    # Fare
    df.Fare = df.Fare.map(lambda x: np.nan if x == 0 else x)
    df.loc[(df.Fare.isnull())&(df.Pclass == 1), 'Fare'] = \
        np.median(df[df['Pclass'] == 1]['Fare'].dropna())
    df.loc[(df.Fare.isnull())&(df.Pclass == 2), 'Fare'] = \
        np.median(df[df['Pclass'] == 2]['Fare'].dropna())
    df.loc[(df.Fare.isnull())&(df.Pclass == 3), 'Fare'] = \
        np.median(df[df['Pclass'] == 3]['Fare'].dropna())

    # creating a title column from name
    title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                'Don', 'Jonkheer']
    df['Title']=df['Name'].map(lambda x: substrings_in_string(x, title_list))

    # Title
    df['Title'] = df.apply(replace_titles, axis=1)
    dummies_Title = pd.get_dummies(df['Title'], prefix='Title')
    df = pd.concat([df, dummies_Title], axis=1)

    # AgeFill
    df['AgeFill'] = df['Age']
    mean_ages = np.zeros(4)
    mean_ages[0] = np.average(df[df['Title'] == 'Miss']['Age'].dropna())
    mean_ages[1] = np.average(df[df['Title'] == 'Mrs']['Age'].dropna())
    mean_ages[2] = np.average(df[df['Title'] == 'Mr']['Age'].dropna())
    mean_ages[3] = np.average(df[df['Title'] == 'Master']['Age'].dropna())
    df.loc[(df.Age.isnull()) & (df.Title == 'Miss'), 'AgeFill'] = mean_ages[0]
    df.loc[(df.Age.isnull()) & (df.Title == 'Mrs'), 'AgeFill'] = mean_ages[1]
    df.loc[(df.Age.isnull()) & (df.Title == 'Mr'), 'AgeFill'] = mean_ages[2]
    df.loc[(df.Age.isnull()) & (df.Title == 'Master'), 'AgeFill'] = mean_ages[3]

    # AgeCat
    df.loc[(df.AgeFill <= 10), 'AgeCat'] = 'child'
    df.loc[(df.AgeFill > 60), 'AgeCat'] = 'aged'
    df.loc[(df.AgeFill > 10) & (df.AgeFill <= 30), 'AgeCat'] = 'adult'
    df.loc[(df.AgeFill > 30) & (df.AgeFill <= 60), 'AgeCat'] = 'senior'
    dummies_AgeCat = pd.get_dummies(df['AgeCat'], prefix='AgeCat')
    df = pd.concat([df, dummies_AgeCat], axis=1)

    # HasCabin
    df["HasCabin"] = 1
    df["HasCabin"][df.Cabin.isnull()] = 0

    # Sex
    le = preprocessing.LabelEncoder()
    le.fit(df['Sex'] )
    x_sex = le.transform(df['Sex'])
    df['Sex'] = x_sex.astype(np.float)

    # Pclass
    dummies_Pclass = pd.get_dummies(df['Pclass'], prefix='Pclass')
    df = pd.concat([df, dummies_Pclass], axis=1)

    # Embarked
    df.Embarked = df.Embarked.fillna('S')
    dummies_Embarked = pd.get_dummies(df['Embarked'], prefix='Embarked')
    df = pd.concat([df, dummies_Embarked], axis=1)

    data = df.loc[:, ['Sex', 'SibSp', 'Parch', 'Fare', 'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs',
                      'AgeCat_adult', 'AgeCat_aged', 'AgeCat_child', 'AgeCat_senior', 'HasCabin',
                      'Pclass_1', 'Pclass_2', 'Pclass_3', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]

    data = preprocessing.scale(data)

    return data