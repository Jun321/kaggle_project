import numpy as np 
import pandas as pd 
import random

from sklearn import cross_validation
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn import ensemble
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, normalize, scale, Normalizer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
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

def get_regression_model(model, length):
	# cv_l = cross_validation.KFold(length, n_folds=10,
	# 							shuffle=True, random_state=1)
	if model == 'LinearRegression':
		regr = LinearRegression()
	elif model == 'RidgeCV':
		regr_ = RidgeCV(cv=5, normalize=True, gcv_mode='auto')
		regr = make_pipeline(PolynomialFeatures(4, interaction_only=True, include_bias=False), regr_)
	elif model == 'GradientBoost':
		params = {'n_estimators': 50, 'max_depth': 4, 'min_samples_split': 1, 'learning_rate': 0.01, 'loss': 'ls'}
		regr = ensemble.GradientBoostingRegressor(**params)
	elif model == 'LassoCV':
		regr_ = LassoCV(cv=5, n_jobs=2, normalize=True, tol=0.0001, max_iter=150000)
		regr = make_pipeline(PolynomialFeatures(4, interaction_only=True, include_bias=False), PCA(n_components=100), regr_)
	elif model == 'KNeighborsRegressor':
		regr = KNeighborsRegressor(21, weights='distance')
	elif model == 'RandomForest':
		regr_ = ensemble.RandomForestRegressor(n_estimators=50, max_depth=4, n_jobs=2, random_state=1, oob_score=True)
		regr = make_pipeline(Normalizer(norm='l2'), regr_)
	elif model == 'SVR':
		regr = SVR()
	else:
		regr = 'NoModel'
	return regr

def get_processed_X(_X):
	X = _X.copy()
	X = X.astype(float)
	# min_max_scaler = MinMaxScaler()
	# X = min_max_scaler.fit_transform(X)
	X = normalize(X, norm='l2')

	return X
def get_processed_X_A(_X):
	X = _X.copy()
	X = X.astype(float)
	min_max_scaler = MinMaxScaler()
	X = min_max_scaler.fit_transform(X)
	# X = scale(X)
	return X

def get_random_trainsets(cv_nm, _df):
	df = _df.copy()
	df = df.iloc[np.random.permutation(len(df))]
	df = df.reset_index(drop = True)
	size = int(np.round(df.shape[0]/cv_nm))
	dfs = [pd.DataFrame(df[(i*size):(i*size+size)]) for i in range(cv_nm)]
	dfs = [df.reset_index(drop = True) for df in dfs]
	return dfs

def get_features():
	features =	[
				'date2j',

				'day',
				'month',
				'year',

				'is_2012', 
				'is_2013', 
				'is_2014',
				'fall', 
				'winter', 
				'spring',
				'summer',

				'weekday',
				'is_weekend', 
				'is_holiday', 
				'is_holiday_weekday', 
				'is_holiday_weekend',

				# 'MemorialDay', 
				# 'MothersDay', 
				'BlackFridayM3',
				'BlackFriday1', 
				# 'NewYearsDay', 
				# 'IndependenceDay', 
				# 'VeteransDay',
				'BlackFriday2', 
				# 'NewYearsEve', 
				'BlackFriday3', 
				# 'ChristmasDay',
				'BlackFridayM2', 
				'ThanksgivingDay', 
				# 'Halloween', 
				# 'EasterSunday',
				'ChristmasEve', 
				# 'ValentinesDay', 
				# 'PresidentsDay', 
				# 'ColumbusDay',
				# 'MartinLutherKingDay', 
				# 'LaborDay', 
				# 'FathersDay', 
				'BlackFriday'
				]

	return features
