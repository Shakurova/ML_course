# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.feature_selection import SelectKBest, f_classif


def read_titanic(filename='titanic.csv'):
	""" Функция читает csv файл"""
	df = pd.read_csv(filename, index_col='PassengerId')
	df.head()

	print(df.describe())
	print("The median age is {} years".format(df['Age'].median()))

	return df


def get_dependence1(df):
	alpha_level = 0.65

	# Sex
	df_sex_survived = df.Sex[df.Survived == 1].value_counts()
	df_sex_survived.plot(kind='bar', alpha=alpha_level)
	plt.title("Distribution of Survival by sex")
	plt.show()
	print('Для пассажира женского пола вероятность выжить была выше, чем у пассажира мужского пола')

	# Pclass
	df_pclass_survived = df.Pclass[df.Survived == 1].value_counts()
	df_pclass_survived.plot(kind='bar', alpha=alpha_level)
	plt.title("Distribution of Survival by Pclass")
	plt.show()
	print(
		'Для пассажира первого класса вероятность выжить была выше, чем у пассажира второго класса. Для пассажира второго'
		'класса вероятность выжить была выше, чем у пассажира первого класса')

	# Fares
	df.Fare[df.Pclass == 1].plot(kind='kde')
	df.Fare[df.Pclass == 2].plot(kind='kde')
	df.Fare[df.Pclass == 3].plot(kind='kde')
	plt.xlabel("Fare")
	plt.title("Fare Distribution within classes")
	plt.legend(('1st Class', '2nd Class', '3rd Class'), loc='best')
	plt.show()


def get_dependence2(df):
	df_male = df.Pclass[df.Sex == 'male'][df.Survived == 1].value_counts()  		# выживших мужчин
	df_female = df.Pclass[df.Sex == 'female'][df.Survived == 1].value_counts()  	# выживших женщин

	df_male.plot(kind='bar', label='Male', alpha=0.70)
	df_female.plot(kind='bar', color='#FA2379', label='Female', alpha=0.70)
	plt.title("Who Survived? with respect to Gender, (raw value counts) ")
	plt.legend(('Male', 'Female'), loc='best')
	plt.show()
	print(
		'У пассажира женского пола первого класса самая высокая вероятность выжить. \nУ женщин первого и второго класса '
		'вероятности выжить больше чем у мужчин их же класса. \nУ мужчин и женщин третьего класса '
		'примерно одинаковая вероятность выжить. ')

	print('P(выжить|женщина, 1 класс) > P(выжить|мужчина, 1 класс)')
	print('P(выжить,женщина) > P(выжить, мужчина)')


def clean_data(df):

	x_labels = ['Pclass', 'Fare', 'Age', 'Sex', 'SibSp', 'Relatives', 'Single', 'Embarked'] 	# параметры
	# Из графиков видно, что пол, класс играют важную роль в решении того, выживет ли человек. Можно также предположить,
	# что наличие семьи тоже играет положительную роль при принятии решении о том, кого посадить вместе

	# df_sex = pd.get_dummies(df['Sex'])
	# df = pd.concat([df, df_sex], axis=1)

	df['Sex'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)  # замена female и male на 0 и 1 соответственно

	embarkments = {"U": 0, "S": 1, "C": 2, "Q": 3}
	df["Embarked"] = df["Embarked"].fillna("U").apply(lambda e: embarkments.get(e))
	print(df["Embarked"])

	df["Relatives"] = df["Parch"] + df["SibSp"]		# новая категория - сколько всего родственников у человека

	df["Single"] = df["Relatives"].apply(lambda r: 1 if r == 0 else 0)   # является ли человек одиночкой

	df = df.dropna()

	X, y = df[x_labels], df['Survived']

	return X, y, x_labels


def train_and_check_model(X_train, X_test, y_train, y_test, model, parameter, x_labels):
	""" Обучение модели и подбор наиболее удачных параментов"""
	clf = model(min_samples_split=parameter)
	clf.fit(np.array(X_train), np.array(y_train))

	importances = pd.Series(clf.feature_importances_, index=x_labels)
	print(importances)

	y_pred = clf.predict(X_test)
	print(classification_report(y_test, y_pred))

	print(np.mean(cross_validation.cross_val_score(clf, X_train, y_train, cv=5)))

	# Выборать наиболее удачный параметр и нарисовать в зависимости от этого дерево
	scores = []
	for t in range(1, parameter):
		rfc = model(min_samples_split=t)
		rfc.fit(X_train, y_train)
		y_pred = rfc.predict(X_test)
		scores.append(f1_score(y_test, y_pred))

	plt.plot(scores)
	plt.xlabel('n_estimators')
	plt.ylabel('score')
	plt.show()


if __name__ == '__main__':

	df = read_titanic()

	print('=' * 60)
	print('1. Зависимость выживания от параметров Sex, Pclass, Fare')
	print('-' * 60)

	get_dependence1(df)

	print('=' * 60)
	print(u'2. Гистограмма, описывающая среднюю вероятность выжить в зависимости от пола и соц. статуса')
	print('-' * 60)

	get_dependence2(df)

	print('=' * 60)
	print(u'2. Очистка данных')
	print('-' * 60)

	X, y, x_labels = clean_data(df)

	print('Подбор наиболее удачного параметра')

	selector = SelectKBest(f_classif, k=5)
	selector.fit(X, y)

	scores = -np.log10(selector.pvalues_)

	plt.bar(range(len(x_labels)), scores)
	plt.xticks(range(len(x_labels)), x_labels, rotation='vertical')
	plt.show()

	print('=' * 60)
	print(u'2. Разделить данные на тестовую и обучающую выборку и обучить DecisionTreeClassifier')
	print('-' * 60)

	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3)
	print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

	train_and_check_model(X_train, X_test, y_train, y_test, DecisionTreeClassifier, 5, x_labels)

	print('=' * 60)
	print(u'2. То же самое для RandomForest')
	print('-' * 60)

	train_and_check_model(X_train, X_test, y_train, y_test, RandomForestClassifier, 100, x_labels)



