import numpy as np 
import pandas as pd 
import util as ut

def test_and_get_res(regrs, tests):
	rmse_total = 0
	se_total = 0
	num_items = 0
	num_test = 0

	for regr, df_test in zip(regrs, tests):
		X_test, y_test = ut.get_train_data(df_test)
		X_test = X_test.drop(['store_nbr','item_nbr'], axis=1)
		y_test = y_test[X_test.index.values]

		X_test = X_test[ut.get_feature_list()]

		prediction = regr.predict(ut.get_processed_X(X_test.values))
		prediction = np.maximum(prediction, 0.)
		rmse = np.sqrt(((y_test.values - prediction) ** 2).mean())
		se = ((y_test.values - prediction) ** 2).sum()

		rmse_total += rmse
		se_total += se
		num_items += 1
		num_test += len(y_test.values)
	# get root-mean-square error total
	print('rmse_test_total: ', rmse_total)
	# get standard deviation total
	print('se_total: ', se_total)
	print('num_items: ', num_items, 'len_of_test: ', num_test)
	print('Average rmse: ', rmse_total / num_items)
	print('Average se: ', se_total / num_test)