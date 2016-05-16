import numpy as np 
import pandas as pd 
import random

from sklearn import cross_validation
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression

def get_train_data(_df):
	df = _df.copy()
	val = df.columns.values.tolist()
	val = [x for x in val if not x in ['date','units', 'date2', 'holiday_name', 'station_nbr']]
	df = df[val]
	y = df['log1p']
	df.drop('log1p', axis=1, inplace=True)
	X = df
	return X, y

def get_test_data(_df):
	df = _df.copy()
	val = df.columns.values.tolist()
	val = [x for x in val if not x in ['units', 'date2', 'holiday_name', 'station_nbr']]
	df = df[val]
	return df

def get_random_test_and_train(_df):
	random.seed(4)
	df = _df.copy()
	rows = random.sample(range(len(df.index)),
			    int(len(df.index) / 5) + 1)
	df_test = df.ix[df.index[rows]]
	df_train = df.drop(df.index[rows])
	return df_test, df_train


def get_feature_list():
	# features = ['date2j', 'weekday', 'day',
	#                    'month', 'year', 'is_weekend',
	#                    'ThanksgivingDay', 'preciptotal', 'tavg',
	#                    'cold', 'fall', 'winter', 
	#                    'spring', 'summer'] 
	features = ['date2j',
	'weekday', 
	'day', 
	'month',
	'is_2012',
	'is_2013',
	'is_2014', 
	'is_weekend', 
	'is_holiday',
	'is_holiday_weekday', 
	'is_holiday_weekend',
	'NewYearsEve', 
	'IndependenceDay', 
	'BlackFridayM3', 
	'BlackFriday',
	'LaborDay', 
	'VeteransDay', 
	'ValentinesDay', 
	'BlackFriday2',
	'ColumbusDay', 
	'FathersDay', 
	'ChristmasEve', 
	'PresidentsDay',
	'BlackFriday1', 
	'NewYearsDay', 
	'MemorialDay', 
	'MothersDay',
	'BlackFridayM2', 
	'ThanksgivingDay', 
	'BlackFriday3', 
	'Halloween',
	'EasterSunday', 
	'MartinLutherKingDay', 
	'high_precip',
	'preciptotal', 
	'snowfall', 
	'high_snow', 
	'avgspeed', 
	'windy',
	'temp_missing', 
	'tavg', 
	'hot', 
	'cold', 
	'frigid', 
	'thunder', 
	'snowcode',
	'raincode', 
	'fall', 
	'winter', 
	'spring', 
	'summer']

	return features

def get_regression_model(model, length):
	cv_l = cross_validation.KFold(length, n_folds=10,
								shuffle=True, random_state=1)
	if model == 'LinearRegression':
		regr = LinearRegression()
	elif model == 'RidgeCV':
		regr = RidgeCV(cv=cv_l)
	elif model == 'SVR':
		regr = SVR()
	elif model == 'LassoCV':
		regr = LassoCV(cv=cv_l, n_jobs=2, normalize=True, tol=0.0001, max_iter=100000)
	elif model == 'KNeighborsRegressor':
		regr = KNeighborsRegressor(21, weights='distance')
	else:
		regr = 'NoModel'
	return regr

def get_processed_X(_X):
	X = _X.copy()
	min_max_scaler = MinMaxScaler()
	X = min_max_scaler.fit_transform(X)
	return X


def get_random_trainsets(cv_nm, _df):
	df = _df.copy()
	df = df.iloc[np.random.permutation(len(df))]
	df = df.reset_index(drop = True)
	size = int(np.round(df.shape[0]/cv_nm))
	dfs = [pd.DataFrame(df[(i*size):(i*size+size)]) for i in range(cv_nm)]
	dfs = [df.reset_index(drop = True) for df in dfs]
	return dfs

