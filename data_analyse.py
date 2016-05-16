import pandas as pd
import numpy as np 

import matplotlib.pyplot as plt
import pylab as p

from errno import EEXIST
from os import makedirs,path

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, f_regression
from sklearn.linear_model import (LinearRegression, Ridge, 
                                  Lasso, RandomizedLasso)
from minepy import MINE

import util as ut


def comprehensive_features_analyse(_df, store_item_nbrs):
	df = _df.copy()
	p.figure()
	X, y = ut.get_train_data(df)
	feature_list = X.columns.values[2:]
	importance_value = np.zeros(len(feature_list))
	total = 0
	for sno, ino in store_item_nbrs:
		if(sno == 35):
			continue
		X_1 = X[(X.store_nbr == sno) & (X.item_nbr == ino)]
		X_1 = X_1.drop(['store_nbr', 'item_nbr'], axis=1)
		y_1 = y[X_1.index.values]
		X_1 = ut.get_processed_X(X_1.values)
		y_1 = y_1.values
		features = feature_list
		train_and_analyse(X_1, y_1, features)

def train_and_analyse(_X, _y, features):
	X = _X
	Y = _y

	ranks = {}

	lr = LinearRegression(normalize=True)
	lr.fit(X, Y)
	ranks["Linear reg"] = rank_to_dict(np.abs(lr.coef_), features)

	ridge = Ridge()
	ridge.fit(X, Y)
	ranks["Ridge"] = rank_to_dict(np.abs(ridge.coef_), features)

	lasso = Lasso(alpha=.005)
	lasso.fit(X, Y)
	ranks["Lasso"] = rank_to_dict(np.abs(lasso.coef_), features)

	rlasso = RandomizedLasso(alpha=.001, max_iter=500000)
	rlasso.fit(X, Y)
	print(rlasso.scores_)
	ranks["Stability"] = rank_to_dict(np.abs(rlasso.scores_), features)

	rfe = RFE(lr, n_features_to_select=5)
	rfe.fit(X,Y)
	ranks["RFE"] = rank_to_dict(np.array(rfe.ranking_).astype(float), features, order=-1)

	rf = RandomForestRegressor()
	rf.fit(X,Y)
	ranks["RF"] = rank_to_dict(rf.feature_importances_, features)

	f, pval  = f_regression(X, Y, center=True)
	ranks["Corr."] = rank_to_dict(np.nan_to_num(f), features)

	mine = MINE()
	mic_scores = []
	for i in range(X.shape[1]):
	    mine.compute_score(X[:,i], Y)
	    m = mine.mic()
	    mic_scores.append(m)
	 
	ranks["MIC"] = rank_to_dict(mic_scores, features) 

	r = {}
	for name in features:
	    r[name] = round(np.mean([ranks[method][name] 
	                             for method in ranks.keys()]), 2)
	 
	methods = sorted(ranks.keys())
	ranks["Mean"] = r
	methods.append("Mean")

	print(pd.DataFrame(ranks))


def f_regression_feature_analyse(_df, store_item_nbrs):
	df = _df.copy()
	p.figure()
	X, y = ut.get_train_data(df)
	feature_list = X.columns.values[2:]
	importance_value = np.zeros(len(feature_list))
	total = 0
	for sno, ino in store_item_nbrs:
		if(sno == 35):
			continue
		X_1 = X[(X.store_nbr == sno) & (X.item_nbr == ino)]
		X_1 = X_1.drop(['store_nbr','item_nbr'], axis=1)
		y_1 = y[X_1.index.values]
		features = feature_list
		F, _ = f_regression(X_1.values, y_1.values)
		importance = get_importance(np.nan_to_num(F))
		print(importance)
		# to draw the each (sno, ino) pic need to uncomment underline code
		# draw_feature_importance(importance, features, sno, ino)
		importance_value += len(X_1.index) * np.array(importance)
		total = total + len(X_1.index)
		print(importance_value)

	importance_value = importance_value / total
	draw_total_average_importance(importance_value, feature_list)

def draw_feature_importance(importance, features, sno, ino):
	p.clf()
	p.cla()
	x = np.arange(np.shape(features)[0]) * 2.5 + .5
	plt.figure(figsize=(16, 12))
	plt.barh(x, importance, align='center', color='c')
	plt.xlabel('Relative Importance')
	plt.yticks(x, features)
	plt.title('Features Importance')
	pic_name = 'Images/Features_Importance/' 
	mkdir_p(pic_name)
	pic_name += 's' + str(sno) + '_i' + str(ino) + '.png'
	p.savefig(pic_name, bbox_inches='tight')
	plt.close()

def draw_total_average_importance(importance_value, feature_list):
	p.clf()
	p.cla()
	x = np.arange(np.shape(feature_list)[0]) * 2.5 + .5
	plt.figure(figsize=(16, 12))
	plt.barh(x, importance_value, align='center', color='c')
	plt.xlabel('Relative Importance')
	plt.yticks(x, feature_list)
	plt.title('Features Importance')
	pic_name = 'Images/Features_Importance/' 
	mkdir_p(pic_name)
	pic_name += 'Total_Importance.png'
	p.savefig(pic_name, bbox_inches='tight')
	plt.close()

def rank_to_dict(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x, 2), ranks)
    return dict(zip(names, ranks ))

def get_importance(F):
	x_1 = np.absolute(F)
	return [x/x_1.sum() for x in x_1]

def mkdir_p(mypath):
	try:
		makedirs(mypath)
	except OSError as exc:
		if exc.errno == EEXIST and path.isdir(mypath):
			pass
		else:
			raise
