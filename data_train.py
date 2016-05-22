import numpy as np 
import pandas as pd
import util as ut
from sklearn.cross_validation import cross_val_score

def train_and_get_result(_df, _dft,  store_item_nbrs, model, total_features):
	df = _df.copy()
	df_t = _dft.copy()
	RES = []
	total = 0
	for sno, ino in store_item_nbrs:
		if(sno == 35):
			continue
		res = pd.DataFrame()
		df1 = df[(df.store_nbr == sno) & (df.item_nbr == ino)]
		X_train, y_train = ut.get_train_data(df1)
		X_train = X_train.drop(['store_nbr', 'item_nbr'], axis=1)
		y_train = y_train[X_train.index.values]

		df2 = df_t[(df_t.store_nbr == sno) & (df_t.item_nbr == ino)]
		X_predict = ut.get_test_data(df2)
		res['date'] = X_predict['date']
		res['store_nbr'] = X_predict['store_nbr']
		res['item_nbr'] = X_predict['item_nbr']
		X_predict = X_predict.drop(['date', 'store_nbr', 'item_nbr'], axis=1)

		X_train = X_train[ut.get_features()]
		X_predict = X_predict[ut.get_features()]
		regr = ut.get_regression_model(model, len(X_train.values))
		regr.fit(X_train.values.astype(float), y_train.values)
		res['log1p'] = np.maximum(regr.predict(X_predict.values.astype(float)), 0.)
		RES.append(res)
		total += 1
		print('done', total)
	result = pd.concat(RES)
	return result

def train_and_get_test(_df, store_item_nbrs, model, total_features):
	df = _df.copy()
	regrs = []
	tests = []
	total = 0
	score_total = []
	for sno, ino in store_item_nbrs:
		if(sno == 35):
			continue
		df1 = df[(df.store_nbr == sno) & (df.item_nbr == ino)]
		df_test, df_train = ut.get_random_test_and_train(df1)
		X_train, y_train = ut.get_train_data(df_train)
		X_train = X_train.drop(['store_nbr', 'item_nbr'], axis=1)
		y_train = y_train[X_train.index.values]

		X_train = X_train[ut.get_features()]
		
		regr = ut.get_regression_model(model, len(X_train))

		# regr.fit(ut.get_processed_X(X_train.values), y_train.values)
		scores = []
		scores = cross_val_score(regr, X_train.values, y_train.values, scoring="mean_squared_error", cv=10)
		print('done, ', total)
		print(-np.mean(scores))
		score_total.append(-np.mean(scores))
		regrs.append(regr)
		tests.append(df_test)
		total += 1
	print('total_score: {}'.format(np.mean(score_total)))

	return regrs, tests

