# -*- coding: utf-8 -*-
import pandas
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

sampleData = data.replace({'Sex': {'male':1,'female':0}}).dropna()

# to the training samples we export only required data, replace string with int and delete rows with NaN value
X = sampleData[['Pclass','Fare','Age','Sex']]

# the class labels for the training samples
Y = sampleData[['Survived']]

clf = tree.DecisionTreeClassifier(random_state=241)
clf = clf.fit(X, Y)

importances = clf.feature_importances_

indices = np.argsort(importances)[::-1]

print(X.columns[indices])

print(importances)

fig, ax = plt.subplots(figsize=(20, 20))
tree.plot_tree(clf)
plt.show()

# Инструкция по выполнению
# Загрузите выборку из файла titanic.csv с помощью пакета Pandas.
# Оставьте в выборке четыре признака: класс пассажира (Pclass), цену билета (Fare), возраст пассажира (Age) и его пол (Sex).
# Обратите внимание, что признак Sex имеет строковые значения.
# Выделите целевую переменную — она записана в столбце Survived.
# В данных есть пропущенные значения — например, для некоторых пассажиров неизвестен их возраст. Такие записи при чтении их в pandas принимают значение nan. Найдите все объекты, у которых есть пропущенные признаки, и удалите их из выборки.
# Обучите решающее дерево с параметром random_state=241 и остальными параметрами по умолчанию (речь идет о параметрах конструктора DecisionTreeСlassifier).
# Вычислите важности признаков и найдите два признака с наибольшей важностью. Их названия будут ответами для данной задачи (в качестве ответа укажите названия признаков через запятую или пробел, порядок не важен).
# Ответ на каждое задание — текстовый файл, содержащий ответ в первой строчке. Обратите внимание, что отправляемые файлы не должны содержать перевод строки в конце. Данный нюанс является ограничением платформы Coursera. Мы работаем над тем, чтобы убрать это ограничение.
