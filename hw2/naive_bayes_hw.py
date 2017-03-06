# -*- coding:utf-8 -*-

"""
SMS Spam/Ham dataset.
1(+6). Проверить, сбалансирован ли датасет (может быть, наблюдений одного класса слишком много?).
Какие результаты покажет dummy classifier, который будет всем новым наблюдениям присваивать класс ham?
Насколько плохо такое решение для задачи определения спама?
Грубое решение - включить в training set только необходимое число наблюдений (примерно поровну spam и ham).
Нормализовать тексты и обучить байесовскую модель (bag of words). Проверить, как влияют на результат:
1) разная токенизация: в одном случае знаки препинания удалять, в другом — считать их токенами;
2) лемматизация (отсутствие лемматизации, стемминг, лемматизация; инструменты можно использовать любые, например, nltk.stem);
!!!!!! 3) удаление стоп-слов, а также пороги минимальной и максимальной document frequency;
4) векторизация документов (CountVectorizer vs. TfIdfVectorizer);
5) что-нибудь ещё?
При оценке классификатора обратите внимание на TP и FP.

Extra: ограничив количество наблюдений ham в обучающей выборке, мы игнорируем довольно много данных. 1)
В цикле: случайно выбрать нужное число писем ham и сконструировать сбалансированную выборку, построить классификатор,
 оценить и записать результат; в итоге результаты усреднить. 2) поможет ли параметр class prior probability?

2(+2). Сравнить результаты байесовского классификатора, решающего дерева и RandomForest. Помимо стандартных метрик
оценки качества модели, необходимо построить learning curve, ROC-curve, classification report и интерпретировать эти результаты.

3(+2). А что, если в качестве предикторов брать не количество вхождений слов, а конструировать специальные признаки?
Прежде всего, необходимо разделить таблицу на training set и test set в соотношении 80:20, test set не открывать до
этапа оценки модели. С помощью pandas проверить, отличаются ли перечисленные ниже параметры (иможно придумать другие)
для разных классов (spam/ham), и собрать матрицу признаков для обучения. Примеры признаков: длина сообщения,
количество букв в ВЕРХНЕМ РЕГИСТРЕ, восклицательных знаков, цифр, запятых, каких-то конкретных слов (для этого
можно построить частотный словарь по сообщениям каждого класса). Прокомментировать свой выбор. Векторизовать
документы и построить классификатор. Оценить модель на проверочной выборке.
"""

import pandas
import numpy as np
import matplotlib.pyplot as plt
from nltk import word_tokenize
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords

from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import fbeta_score, make_scorer
from sklearn import cross_validation
from sklearn.base import TransformerMixin

from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_curve, auc

stop_words = stopwords.words('russian')


def tokenize(text):
	""" Токенизация, знаки препинания удаляются """
	text = text.lower()
	return word_tokenize(text)


def tokenize2(text):
	""" Токенизация, знаки препинания оставляются """
	text = text.lower()
	return wordpunct_tokenize(text)


def do_smth_with_model(steps):
	""" """
	print('\nModel train')
	pipeline = Pipeline(steps=steps)

	cv_results = cross_val_score(pipeline,
								 msg_train,
								 label_train,
								 cv=10,
								 scoring='accuracy',
								 )
	print(cv_results.mean(), cv_results.std())

	pipeline.fit(msg_train, label_train)
	label_predicted = pipeline.predict(msg_test)
	print(label_predicted)

	print(classification_report(label_test, label_predicted ))

	return pipeline, label_predicted


def draw_learning_curve(pipeline):
	print('Draw learning curve')
	train_sizes, train_scores, valid_scores = learning_curve(pipeline, msg_train, label_train,
															 train_sizes=[50, 80, 110], cv=5)
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(valid_scores, axis=1)
	test_scores_std = np.std(valid_scores, axis=1)

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
	plt.show()


def draw_roc_curve(label_predicted):
	print('Draw roc curve')
	false_positive_rate, true_positive_rate, thresholds = roc_curve(label_test, label_predicted)
	roc_auc = auc(false_positive_rate, true_positive_rate)
	plt.title('Receiver Operating Characteristic')
	plt.plot(false_positive_rate, true_positive_rate, 'b',
			 label='AUC = %0.2f' % roc_auc)
	plt.legend(loc='lower right')
	plt.plot([0, 1], [0, 1], 'r--')
	plt.xlim([-0.1, 1.2])
	plt.ylim([-0.1, 1.2])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.show()


def length(text):
	"""Длина текста."""
	return len(text)


def uppercase(text):
	"""Сколько заглавных букв."""
	return len([letter for letter in list(text) if letter.isupper()])


def exclamation(text):
	"""Сколько !."""
	return len([letter for letter in list(text) if letter == '!'])


def numbers(text):
	"""Сколько чисел."""
	return len([letter for letter in list(text) if letter.isdigit()])


def warning_words(text):
	"""Сколько слов, типичных для спам-сообщений."""
	return len([word for word in tokenize(text) if word in spam_words])


class FunctionFeaturizer(TransformerMixin):
	""" Для создания своего вектора я использовала несколько фич: длину текста, количество заглавных букв
	 (чем больше, тем обычно выше вероятность, что это спам), количество ! (в спам-сообщениях встречаются часто),
	 количество чисел, сколько слов из словаря спам-слов (50 самых частых слов в коллекции спам-сообщений)"""
	def __init__(self, *featurizers):
		self.featurizers = featurizers

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		fvs = []
		for datum in X:
			fv = [f(datum) for f in self.featurizers]
			fvs.append(fv)
		return np.array(fvs)

if __name__ == '__main__':

	spam_featurizer = FunctionFeaturizer(length, uppercase, exclamation, numbers,
										warning_words)  # создание своего векторизатора

	path = 'smsspamcollection/SMSSpamCollection'

	# Чтение из файла
	messages = pandas.read_csv(path, sep='\t',
							   names=["label", "message"])

	messages['length'] = messages['message'].map(lambda text: len(text))
	print(messages.head())

	# Ститастика по словам, которые встречаются в спам-сообщениях
	arr = {}
	for m in messages.message[messages.label=='spam']:
		for word in tokenize(m):
			if word not in arr:
				arr[word] = 1
			else:
				arr[word] += 1
	spam_words = []
	for key in sorted(arr.items(), key=lambda x:x[1], reverse=True)[:50]:
		# print(key[0])
		spam_words.append(key[0])  # словарь спам-слов

	# Проверить, сбалансирован ли data set:
	spam = 0
	ham = 0
	for i in messages['label']:
		if i == 'ham':
			ham += 1
		if i == 'spam':
			spam += 1
	print(ham)
	print(spam)

	print(messages.groupby('label').describe())

	print('\nВыборка несбалансированна, неспам - 4825, спам - 747, примеров спама гораздо меньше')

	# перевод str в int (ham->0, spam->1)
	messages['label'] = messages['label'].map({'ham': 0, 'spam': 1}).astype(int)

	# Векторизация
	bow = CountVectorizer()
	bow.fit_transform(messages['message'])
	bowed_messages = bow.transform(messages['message'])

	# Обучение DummyClassifier
	clf = DummyClassifier(strategy='most_frequent', random_state=0)
	clf = clf.fit(bowed_messages, messages['label'])

	# Вывод результатов по Dummy Classifier
	print(classification_report(messages['label'], clf.predict(bowed_messages)))
	print('Dummy classifier, который будет всем новым наблюдениям присваивать класс ham, получит 75% precission и 87 -  recall, 80 - f-score')

	# print('\nNaive Bayes 1')
	# naive_model = MultinomialNB()
	# naive_model.fit(bowed_messages, messages['label'])
	# # print(len(msg_train), len(msg_test))
	# cv_results = cross_val_score(naive_model, bowed_messages, messages['label'], cv=10, scoring='accuracy')
	# print(cv_results.mean(), cv_results.std())
	# print(classification_report(messages['label'], naive_model.predict(bowed_messages)))

	msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.2) # поделить выборку в соотновении 80:20

	# Первая токенизация, Байес
	print('\n1) Naive Bayes,  tokenize 1')
	pipeline, label_predicted = do_smth_with_model(steps=[('bow', CountVectorizer(analyzer=tokenize)),
							  ('classifier', MultinomialNB())])

	draw_learning_curve(pipeline)
	draw_roc_curve(label_predicted)
	print('Судя по roc-curve, классификатор показывает высокие результаты, AUC-value очень высокий, roc-curve почти параллельна оси х')
	print('Learning curve показывает, что при увеличении обучающих данных, cross-validation score может незначительно '
		  'улучшиться, training score при этом останется статичен')


	# Вторая токенизация, Байес
	print('\n1) Naive Bayes tokenize 2')
	do_smth_with_model(steps=[('bow', CountVectorizer(analyzer=tokenize2)),
							  ('classifier', MultinomialNB())])

	# Первая токенизация, Байес, удаляем стоп слова
	print('\n3) Naive Bayes удаляем стоп слова')
	do_smth_with_model(steps=[('bow', CountVectorizer(analyzer=tokenize, stop_words=stop_words)),
							  ('classifier', MultinomialNB())])

	# Байес, векторизация tf-idf
	print('\n4) Векторизация tf-idf')
	do_smth_with_model(steps=[('vect', CountVectorizer()),
							  ('tfidf', TfidfTransformer()),
							  ('classifier', MultinomialNB())])

	# Байес, векторизация tf-idf, fit_prior=False
	print('\nExtra Векторизация tf-idf, fit_prior=False')
	do_smth_with_model(steps=[('vect', CountVectorizer()),
							  ('tfidf', TfidfTransformer()),
							  ('classifier', MultinomialNB(fit_prior=False))])

	# Дерево принятий решений, tf-idf
	print('\nDecission Tree')
	pipeline, label_predicted = do_smth_with_model(steps=[('vect', CountVectorizer()),
							  ('tfidf', TfidfTransformer()),
							  ('classifier', DecisionTreeClassifier())])

	draw_learning_curve(pipeline)
	draw_roc_curve(label_predicted)
	print('Learning curve показывает, что при увеличении обучающих данных, cross-validation score может незначительно '
		  'улучшиться, training score при этом останется статичен')
	print('Судя по roc-curve, классификатор показывает высокие результаты, но у наивного байесы было лучше '
		  '(надо смотреть на наклон синей прямой). AUC-value хуже, чем у Байеса, лучше, чем у случайного леса ')

	# Случайный лес, tf-idf
	print('\nRandomForestClassifier')
	pipeline, label_predicted = do_smth_with_model(steps=[('vect', CountVectorizer()),
							  ('tfidf', TfidfTransformer()),
							  ('classifier', RandomForestClassifier())])

	draw_learning_curve(pipeline)
	draw_roc_curve(label_predicted)
	print('Learning curve показывает, что при увеличение обучающих данных практически ничего не даст')
	print('AUC-value равен достаточно высокий, но хуже, чем у наивного байеса и случайного леса')

	# Свой векторизатор
	print('\nCustom Transformer')
	pipeline, label_predicted = do_smth_with_model(steps=[('custom', spam_featurizer),
							  ('classifier', MultinomialNB())])


	print('Все вышеописанные классификаторы дают достаточно высокие результаты. fit_prior=False результаты не улучшает.'
		  'У Random forest самый высокий Precision у 1. Наивный байес дал лучшие результаты, чем случайный лес и дерево решений')
