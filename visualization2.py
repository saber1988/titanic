# -*- coding: utf-8 -*-
__author__ = 'shidaiting01'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_train = pd.read_csv("data/train.csv")

data_train.info()
data_train.describe()

fig = plt.figure()
fig.set(alpha=.5)  # 设定图表颜色alpha参数

plt.subplot2grid((3, 3), (0, 0))             # 在一张大图里分列几个小图
data_train.Survived.value_counts().plot(kind="bar")
plt.ylabel("passenger")
plt.title("Survived")

plt.subplot2grid((3, 3), (0, 1))
data_train.Pclass.value_counts().plot(kind="bar")
plt.title("Pclass")

plt.subplot2grid((3, 3), (0, 2))
plt.scatter(data_train.Survived, data_train.Age)                         # sets the y axis lable
plt.grid(b=True, which='major', axis='y') # formats the grid line style of our graphs
plt.title("Age")

plt.subplot2grid((3, 3), (1, 0), colspan=2)
data_train.Age[data_train.Pclass == 1].plot(kind='kde')   # plots a kernel desnsity estimate of the subset of the 1st class passanges's age
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel("Age")# plots an axis lable
plt.title("Age distribution for Pclass")
plt.legend(('1', '2', '3'),loc='best') # sets our legend for our graph.

plt.subplot2grid((3, 3), (1, 2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title("Embarked")

plt.subplot2grid((3, 3), (2, 0))
plt.title("Age")
data_train['Age'].dropna().hist(bins=16, range=(0, 80), alpha = .5)
plt.savefig("plot/histogram1.png")

def proportionSurvived(discreteVar):
    by_var = data_train.groupby([discreteVar,'Survived'])
    table = by_var.size().unstack()
    normedtable = table.div(table.sum(1), axis=0)
    return normedtable

discreteVarList = ['Sex', 'Pclass', 'Embarked']

fig1, axes1 = plt.subplots(3,1)

for i in range(3):
    var = discreteVarList[i]
    table = proportionSurvived(var)
    table.plot(kind='barh', stacked=True, ax=axes1[i])
fig1.savefig("plot/histogram2.png")

fig2, axes2 = plt.subplots(2,3)
genders=data_train.Sex.unique()
classes=data_train.Pclass.unique()

#normalise rgb
def normrgb(rgb):
    rgb = [float(x)/255 for x in rgb]
    return rgb


#list our colors
darkpink, lightpink =normrgb([255,20,147]), normrgb([255,182,193])
darkblue, lightblue = normrgb([0,0,128]),normrgb([135,206, 250])
for gender in genders:
    for pclass in classes:
        if gender=='male':
            colorscheme = [lightblue, darkblue]
            row=0
        else:
            colorscheme = [lightpink, darkpink]
            row=1
        group = data_train[(data_train.Sex==gender)&(data_train.Pclass==pclass)]
        group = group.groupby(['Embarked', 'Survived']).size().unstack()
        group = group.div(group.sum(1), axis=0)
        group.plot(kind='barh', ax=axes2[row, (int(pclass)-1)], color=colorscheme, stacked=True, legend=False).set_title('Class '+str(pclass)).axes.get_xaxis().set_ticks([])

plt.subplots_adjust(wspace=0.4, hspace=1.3)

fhandles, flabels = axes2[1,2].get_legend_handles_labels()
mhandles, mlabels = axes2[0,2].get_legend_handles_labels()
plt.figlegend(fhandles, ('die', 'live'), title='Female', loc='center', bbox_to_anchor=(0.06, 0.45, 1.1, .102))
plt.figlegend(mhandles, ('die', 'live'), 'center', title='Male',bbox_to_anchor=(-0.15, 0.45, 1.1, .102))

fig2.savefig("plot/histogram3.png")

def removeBadStringFromString(string, badStringList):
    for badString in badStringList:
        string = string.replace(badString, '')
    return string

def removeBadStringFromLabels(ax, badStringList):
    labels = [item.get_text() for item in ax.get_yticklabels()]
    labels = [removeBadStringFromString(label, badStringList) for label in labels]
    return labels

def plotVarsGroupedBySex(varList, varNames, badStringList, num):
    fig, axes = plt.subplots(1,2)
    for i in range(2):
        group = data_train.groupby([varList[i], 'Sex', 'Survived'])
        group = group.size().unstack()
        group = group.div(group.sum(1), axis=0)
        cols = [[darkpink, darkblue],[lightpink, lightblue]]
        group.plot(kind='barh', stacked=True, ax=axes[i],legend=False, color=cols)
        labels = removeBadStringFromLabels(axes[i], badStringList)
        axes[i].set_yticklabels(labels)
        axes[i].get_xaxis().set_ticks([])
        axes[i].set_ylabel('')
        axes[i].set_title(varNames[i])

        if i ==1:
            axes[i].yaxis.tick_right()
            axes[i].yaxis.set_label_position("right")

    handles, labels = axes[0].get_legend_handles_labels()
    plt.figlegend(handles[0], ['die', 'die'], loc='lower left')
    plt.figlegend(handles[1], ['live','live'], loc='lower right')
    fig.savefig("plot/histogram{}.png".format(num))

bins = [0,5,14, 25, 40, 60, 100]
binNames =['Young Child', 'Child', 'Young Adult', 'Adult', 'Middle Aged', 'Older']
binAge = pd.cut(data_train.Age, bins, labels=binNames)
binFare = pd.qcut(data_train.Fare, 3, labels=['Cheap', 'Medium', 'Expensive'])
binVars = [binAge, binFare]
varNames = ['Age', 'Fare']
badStringList=['(', ')', 'female', 'male', ',']
plotVarsGroupedBySex(binVars, varNames, badStringList, 4)

varNames = ['# Parents or Children', '# Siblings or Spouses']
varList = ['Parch', 'SibSp']
plotVarsGroupedBySex(varList, varNames, badStringList, 5)

fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df=pd.DataFrame({'survived':Survived_1, 'not survived':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title("Pclass")

plt.savefig("plot/histogram6.png")

fig=plt.figure()
fig.set(alpha=0.65) # 设置图像透明度，无所谓
plt.title("Pclass and Sex")

ax1=fig.add_subplot(141)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar', label="female highclass", color='#FA2479')
ax1.set_xticklabels(['survived', 'not survived'], rotation=0)
ax1.legend(["female high"], loc='best')

ax2=fig.add_subplot(142, sharey=ax1)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='female, low class', color='pink')
ax2.set_xticklabels(['survived', 'not survived'], rotation=0)
plt.legend(["female low"], loc='best')

ax3=fig.add_subplot(143, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar', label='male, high class',color='lightblue')
ax3.set_xticklabels(['survived', 'not survived'], rotation=0)
plt.legend(["male high"], loc='best')

ax4=fig.add_subplot(144, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='male low class', color='steelblue')
ax4.set_xticklabels(['survived', 'not survived'], rotation=0)
plt.legend(["male low"], loc='best')
plt.savefig("plot/histogram7.png")

fig = plt.figure()
g = data_train.groupby(['SibSp','Survived'])
df = pd.DataFrame(g.count()['PassengerId']).unstack()
df.plot(kind='bar', stacked=False)
plt.title("SibSp")
plt.savefig("plot/histogram8.png")


fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
df=pd.DataFrame({'has':Survived_cabin, 'hasn\'t':Survived_nocabin})
df.plot(kind='bar', stacked=True)
plt.title("Cabin")
plt.savefig("plot/histogram9.png")