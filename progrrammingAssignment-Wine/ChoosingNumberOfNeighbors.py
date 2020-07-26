# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale

data = np.loadtxt('wine.data', delimiter=",")
X = data[:, 1:14]
y = data[:, 0]

# in the documentation for model_selected.StrafiedKFold, there is no keyword argument called n_folds and you should indeed use n_splits.
kf = KFold(len(X), n_splits=5, shuffle=True, random_state=42)

kMeans = list()
for k in range(1, 51):
    # Технически кросс-валидация проводится в два этапа:
    #
    # Создается генератор разбиений sklearn.model_selection.KFold, который задает набор разбиений на обучение и валидацию.
    # Число блоков в кросс-валидации определяется параметром n_splits. Обратите внимание, что порядок следования объектов
    # в выборке может быть неслучайным, это может привести к смещенности кросс-валидационной оценки. Чтобы устранить такой
    # эффект, объекты выборки случайно перемешивают перед разбиением на блоки. Для перемешивания достаточно передать генератору
    # KFold параметр shuffle=True.
    # Вычислить качество на всех разбиениях можно при помощи функции sklearn.model_selection.cross_val_score.
    # В качестве параметра estimator передается классификатор, в качестве параметра cv — генератор разбиений с предыдущего шага.
    # С помощью параметра scoring можно задавать меру качества, по умолчанию в задачах классификации используется доля верных ответов
    # (accuracy). Результатом является массив, значения которого нужно усреднить.
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X, y)
    array = cross_val_score(estimator=neigh, X=X, y=y, cv=kf, scoring='accuracy')
    m = array.mean()
    kMeans.append(m)

m = max(kMeans)
indices = [i for i, j in enumerate(kMeans) if j == m]

print(indices[0] + 1)  # 1
print(np.round(m, decimals=2))  # 2

x_scale = scale(X)

kf = KFold(len(x_scale), n_splits=5, shuffle=True, random_state=42)

kMeans = list()
for k in range(1, 51):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(x_scale, y)
    array = cross_val_score(estimator=neigh, X=x_scale, y=y, cv=kf, scoring='accuracy')
    m = array.mean()
    kMeans.append(m)

m = max(kMeans)
print(kMeans)
indices = [i for i, j in enumerate(kMeans) if j == m]
print(indices)

print(indices[0] + 1)  # 3
print(np.round(m, decimals=2))  # 4

# Инструкция по выполнению
# В этом задании вам нужно подобрать оптимальное значение k для алгоритма kNN. Будем использовать набор данных Wine,
# где требуется предсказать сорт винограда, из которого изготовлено вино, используя результаты химических анализов.
#
# Выполните следующие шаги:
#
# 1) Загрузите выборку Wine по адресу https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data (файл также приложен к этому заданию)
# 2) Извлеките из данных признаки и классы. Класс записан в первом столбце (три варианта), признаки — в столбцах со второго по последний.
# Более подробно о сути признаков можно прочитать по адресу https://archive.ics.uci.edu/ml/datasets/Wine
# (см. также файл wine.names, приложенный к заданию)
# 3) Оценку качества необходимо провести методом кросс-валидации по 5 блокам (5-fold). Создайте генератор разбиений,
# который перемешивает выборку перед формированием блоков (shuffle=True). Для воспроизводимости результата, создавайте
# генератор KFold с фиксированным параметром random_state=42. В качестве меры качества используйте долю верных ответов (accuracy).
# 4) Найдите точность классификации на кросс-валидации для метода k ближайших соседей (sklearn.neighbors.KNeighborsClassifier),
# при k от 1 до 50. При каком k получилось оптимальное качество? Чему оно равно (число в интервале от 0 до 1)?
# Данные результаты и будут ответами на вопросы 1 и 2.
# 5) Произведите масштабирование признаков с помощью функции sklearn.preprocessing.scale. Снова найдите оптимальное k на кросс-валидации.
# 6) Какое значение k получилось оптимальным после приведения признаков к одному масштабу? Приведите ответы на вопросы 3 и 4. Помогло ли масштабирование признаков?
