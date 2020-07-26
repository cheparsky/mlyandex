# -*- coding: utf-8 -*-
import pandas
import numpy as np
data = pandas.read_csv('titanic.csv', index_col='PassengerId')

#1
data['Sex'].value_counts()

#2
data.groupby('Survived').count() / len(data) * 100

#3
data.groupby('Pclass').count() / len(data) * 100

#4
middle = data['Age'].mean()
median = data['Age'].median()

#5
print data.SibSp.corr(data.Parch)