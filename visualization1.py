__author__ = 'shidaiting01'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing


def substrings_in_string(big_string, substrings):
    import string
    for substring in substrings:
        if string.find(big_string, substring) != -1:
            return substring
    print big_string
    return np.nan

train = pd.read_csv("data/train.csv", dtype={"Age": np.float64}, )

train['Age'].dropna().hist(bins=16, range=(0,80), alpha = .5)
#plt.show()
plt.savefig("plot/histogram.png")

#train["Gender"] = 0
#train["Gender"][train["Sex"] == 'female'] = 1
train['Gender'] = train['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

train["HasCabin"] = 1

train.Fare = train.Fare.map(lambda x: np.nan if x==0 else x)
train.loc[ (train.Fare.isnull())&(train.Pclass==1),'Fare'] =np.median(train[train['Pclass'] == 1]['Fare'].dropna())
train.loc[ (train.Fare.isnull())&(train.Pclass==2),'Fare'] =np.median( train[train['Pclass'] == 2]['Fare'].dropna())
train.loc[ (train.Fare.isnull())&(train.Pclass==3),'Fare'] = np.median(train[train['Pclass'] == 3]['Fare'].dropna())

#creating a title column from name
title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
            'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
            'Don', 'Jonkheer']
train['Title']=train['Name'].map(lambda x: substrings_in_string(x, title_list))

#replacing all titles with mr, mrs, miss, master
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
    elif title =='':
        if x['Sex']=='Male':
            return 'Master'
        else:
            return 'Miss'
    else:
        return title

train['Title']=train.apply(replace_titles, axis=1)
dummies_Title = pd.get_dummies(train['Title'], prefix= 'Title')
train = pd.concat([train, dummies_Title], axis=1)

train["Age"][np.isnan(train["Age"])] = np.median(train["Age"])
train['AgeFill']=train['Age']
mean_ages = np.zeros(4)
mean_ages[0]=np.average(train[train['Title'] == 'Miss']['Age'].dropna())
mean_ages[1]=np.average(train[train['Title'] == 'Mrs']['Age'].dropna())
mean_ages[2]=np.average(train[train['Title'] == 'Mr']['Age'].dropna())
mean_ages[3]=np.average(train[train['Title'] == 'Master']['Age'].dropna())
train.loc[ (train.Age.isnull()) & (train.Title == 'Miss') ,'AgeFill'] = mean_ages[0]
train.loc[ (train.Age.isnull()) & (train.Title == 'Mrs') ,'AgeFill'] = mean_ages[1]
train.loc[ (train.Age.isnull()) & (train.Title == 'Mr') ,'AgeFill'] = mean_ages[2]
train.loc[ (train.Age.isnull()) & (train.Title == 'Master') ,'AgeFill'] = mean_ages[3]

train['AgeCat']=train['AgeFill']
train.loc[ (train.AgeFill<=10) ,'AgeCat'] = 'child'
train.loc[ (train.AgeFill>60),'AgeCat'] = 'aged'
train.loc[ (train.AgeFill>10) & (train.AgeFill <=30) ,'AgeCat'] = 'adult'
train.loc[ (train.AgeFill>30) & (train.AgeFill <=60) ,'AgeCat'] = 'senior'

train["Family_Size"] = train['SibSp'] + train['Parch']
train['Family'] = train['SibSp'] * train['Parch']

train['Fare_Per_Person']=train['Fare']/(train['Family_Size']+1)

train['AgeClass']=train['AgeFill'] * train['Pclass']
train['ClassFare'] = train['Pclass'] * train['Fare_Per_Person']


train['HighLow']=train['Pclass']
train.loc[(train.Fare_Per_Person < 8), 'HighLow'] = 'Low'
train.loc[(train.Fare_Per_Person >= 8), 'HighLow'] = 'High'

train["HasCabin"][train.Cabin.isnull()] = 0

train.Embarked = train.Embarked.fillna('S')

le = preprocessing.LabelEncoder()
le.fit(train['Sex'] )
x_sex=le.transform(train['Sex'])
train['Sex']=x_sex.astype(np.float)

le.fit( train['Ticket'])
x_Ticket=le.transform( train['Ticket'])
train['Ticket']=x_Ticket.astype(np.float)

le.fit(train['Title'])
x_title=le.transform(train['Title'])
train['Title'] =x_title.astype(np.float)

le.fit(train['HighLow'])
x_hl=le.transform(train['HighLow'])
train['HighLow']=x_hl.astype(np.float)


le.fit(train['AgeCat'])
x_age=le.transform(train['AgeCat'])
train['AgeCat'] =x_age.astype(np.float)

le.fit(train['Embarked'])
x_emb=le.transform(train['Embarked'])
train['Embarked']=x_emb.astype(np.float)


plt.figure(figsize=(12, 10))
_ = sns.corrplot(train, annot=False)

#plt.show()
plt.savefig("plot/corr_plot.png")

# Replacing missing ages with median
train["Survived"][train["Survived"]==1] = "Survived"
train["Survived"][train["Survived"]==0] = "Died"
train["ParentsAndChildren"] = train["Parch"]
train["SiblingsAndSpouses"] = train["SibSp"]

plt.figure()
sns.pairplot(data=train[["Fare","Survived","Age","ParentsAndChildren","SiblingsAndSpouses","Pclass","Gender","HasCabin"]],
             hue="Survived", dropna=True)
plt.savefig("plot/pair_plot.png")

