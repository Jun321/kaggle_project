import numpy as np
import pandas as pd

import data_clean as dc
import data_preprocess as dp
import data_analyse as da
import data_train as dtr
import data_test as dte
import data_submission as ds
import util as ut
import timeit


# load specific data
tic0 = timeit.default_timer()
df_train = pd.read_csv("data/train.csv")
wtr = pd.read_csv("data/weather.csv")
key = pd.read_csv('data/key.csv')
df_test = pd.read_csv('data/test.csv')
tic1 = timeit.default_timer()
print('Load Time', tic1 - tic0)
#---------------------------------------------------------
flag_for_test = False

wtr = dc.weather_process(wtr)

if flag_for_test:
	dfs = ut.get_random_trainsets(5, df_train)
	for df_train in dfs:
		store_item_nbrs = dc.create_vaild_item_store_combinations(df_train)
		df_train = dc.get_filted_data(df_train, store_item_nbrs, True)
		df_train = dp.get_preprocessed_data(df_train, wtr, key)
		total_features = da.comprehensive_features_analyse(df_train, store_item_nbrs)
		tic3 = timeit.default_timer()
		regrs, tests = dtr.train_and_get_test(df_train, store_item_nbrs, 'KNeighborsRegressor', total_features)
		dte.test_and_get_res(regrs, tests, total_features)
		tic4_1 = timeit.default_timer()
		print('Train Time', tic4_1 - tic3)
else:
	# exclude the data which in the pair of (sno, ino) to group
	# and the value of log1p_unit is 0
	# in result there are 255 pair of (sno, ino)
	store_item_nbrs = dc.create_vaild_item_store_combinations(df_train)
	#---------------------------------------------------------

	# remain the train data set in store_item_nbrs
	# add log1p for the units
	# -----------NOTE!!!! date('2013-12-25') need to be excluded
	# because of the day sales only little units
	df_train = dc.get_filted_data(df_train, store_item_nbrs, True)
	df_test = dc.get_filted_data(df_test, store_item_nbrs, False)
	#---------------------------------------------------------

	# select and add feature of weather which is 
	# 'high_precip': 1 for precip >= 0.75 
	# 'preciptotal': remain precip and clean it
	# 'snowfall': remain snowfall and clean it
	# 'high_snow': 1 for snowfall >= 1.5
	# 'avgspeed': remain avgspeed and clean it
	# 'windy': 1 for avgspeed >= 18
	# 'temp_missing': 1 for tavg == 'M'
	# 'tavg': remain tavg and clean it
	# 'hot': 1 for tavg >= 80
	# 'cold': 1 for tavg <= 32
	# 'frigid': 1 for tavg <= 15
	# 'thunder': 1 for codesum contain 'TS'
	# 'snowcode': 1 for codesum contain 'SN'
	# 'raincode': 1 for codesum contain 'RA'
	# ------------NOTE!!!! There is no station 5 in the list
	# but it doesnt matter as the test set dont have it (5->store_nbr(35))
	# wtr = dc.weather_process(wtr)
	tic2 = timeit.default_timer()
	print('Clean Time', tic2 - tic1)
	#-------------------------------------------------------------

	# merge the wtr and df_train 
	# create feature about datatime
	# create feature about weather
	# datatime:
	# 'date2j': delta days to 2012-01-01
	# 'weekday': day of week
	# 'day': day of year
	# 'month': month of year
	# 'is_weekend': 1 for is in weekend
	# 'is_holiday': 1 for is in holiday
	# 'is_holiday_weekday': 1 for is in holiday and weekday
	# 'is_holiday_weekend': 1 for is in holiday and weekend
	# 'XXXX(holiday_name)': 1 for is in this holiday
	# weather:
	# 'fall': is in fall
	# 'winter': is in winter
	# 'spring': is in spring
	# 'summer': is in summer
	# ------------NOTE!!!! IT HAVE BEEN EXCLUDE THE DATASET(store_nbr == 35)
	df_train = dp.get_preprocessed_data(df_train, wtr, key)
	df_test = dp.get_preprocessed_data(df_test, wtr, key)
	tic3 = timeit.default_timer()
	print('Preprocess Time', tic3 - tic2)
	#-------------------------------------------------------------

	# analyse the features use f_regression(module of sklearn)
	# http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html#sklearn.feature_selection.f_regression
	# get score of each and draw the png for each importance of (sno, ino) dataset
	# evaluate the importance and find the date give the more importance than weather
	# ------------To Run need to uncomment the underline code
	#
	total_features, total_coef = da.comprehensive_features_analyse(df_train, store_item_nbrs)
	print(total_features)
	#
	#-------------------------------------------------------------

	# 2 choice:
	# 1: take 1/5 train set to test and get
	# rmse_test_total, se_total and average of them
	# to test model and feature selection
	# 2: train the all data set and get test score from kaggle
	# RMSLE: https://www.kaggle.com/c/walmart-recruiting-sales-in-stormy-weather/details/evaluation
	# select the model to run and check score in choice 1
	# find knnRegre have a good score
	# RES = dtr.train_and_get_result(df_train, df_test, store_item_nbrs, 'KNeighborsRegressor', total_features)
	# tic4 = timeit.default_timer()
	# print('Train Time', tic4 - tic3)
	# ------------To Run TEST need to uncomment the underline 2 codes
	regrs, tests = dtr.train_and_get_test(df_train, store_item_nbrs, 'SVR', total_features, total_coef)
	dte.test_and_get_res(regrs, tests, total_features, total_coef)
	tic4_1 = timeit.default_timer()
	# print('Train Time', tic4_1 - tic3)
	#-------------------------------------------------------------

	# after submission:
	# knn have a score of 0.09758 and in 31 place
	# the other will provided after check and improvement
	# ds.submission_to_file('submission_KNeighborsRegressor', RES, store_item_nbrs)
	# tic5 = timeit.default_timer()
	# print('Submission Time', tic5 - tic4)
	# print('Total Time', tic5 - tic0)

