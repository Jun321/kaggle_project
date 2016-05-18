import numpy as np 
import pandas as pd 
import random

from sklearn import cross_validation
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn import ensemble
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
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

def get_regression_model(model, length, coef):
	rng = check_random_state(0)
	cv_l = cross_validation.KFold(length, n_folds=10,
								shuffle=True, random_state=1)
	if model == 'LinearRegression':
		regr = LinearRegression()
	elif model == 'RidgeCV':
		regr = RidgeCV(cv=cv_l)
	elif model == 'GradientBoost':
		params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 1, 'learning_rate': 0.01, 'loss': 'ls'}
		regr = ensemble.GradientBoostingRegressor(**params)
	elif model == 'LassoCV':
		regr = LassoCV(cv=cv_l, n_jobs=2, normalize=True, tol=0.0001, max_iter=100000)
	elif model == 'KNeighborsRegressor':
		regr = KNeighborsRegressor(21, weights='distance')
	elif model == 'RandomForest':
		regr = ensemble.RandomForestRegressor(n_estimators=200, max_depth=4, n_jobs=2)
	elif model == 'SVR':
		grid = ParameterGrid({"max_samples": [0.5, 1.0],
						  "max_features": [0.5, 1.0],
						  "bootstrap": [True, False],
						  "bootstrap_features": [True, False]})

		for base_estimator in [None,
							   DummyRegressor(),
							   DecisionTreeRegressor(),
							   KNeighborsRegressor(),
							   SVR()]:
			for params in grid:
				ensemble.BaggingRegressor(base_estimator=base_estimator,
								 random_state=rng,
								 **params)
		regr = SVR()
	else:
		regr = 'NoModel'
	return regr

def get_processed_X(_X):
	X = _X.copy()
	X = X.astype(float)
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

