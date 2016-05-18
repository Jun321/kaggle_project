import pandas as pd
import numpy as np 

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import pylab as p

from errno import EEXIST
from os import makedirs,path

from sklearn.preprocessing import MinMaxScaler
from sklearn import cross_validation
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, f_regression
from sklearn.linear_model import (LinearRegression, LassoCV, RidgeCV, RandomizedLasso)
from minepy import MINE

import util as ut


def comprehensive_features_analyse(_df, store_item_nbrs):
	df = _df.copy()
	X, y = ut.get_train_data(df)
	feature_list = X.columns.values[2:]
	importance_value = np.zeros(len(feature_list))
	total_weight = 0
	total_rank = pd.DataFrame()
	total_features = []
	total_coef = []
	total = 0
	for sno, ino in store_item_nbrs:
		if(sno == 35):
			continue
		X_1 = X[(X.store_nbr == sno) & (X.item_nbr == ino)]
		X_1 = X_1.drop(['store_nbr', 'item_nbr'], axis=1)
		y_1 = y[X_1.index.values]
		X_1 = ut.get_processed_X(X_1.values)
		y_1 = y_1.values
		weight = y_1[y_1 > 0].shape[0]
		total_weight += weight
		features = feature_list
		rank, selection_feature, coef = train_and_analyse(X_1, y_1, features)
		if(len(total_rank) == 0):
			total_rank = rank
		total_features.append(selection_feature)
		total_coef.append(coef)
		total_rank += rank * weight
		total += 1
		print('done', total)
	# total_rank /= total_weight
	# total_rank.plot.barh(stacked=True)
	# total_rank.to_pickle('total_rank')
	# plt.show()
	# plt.close()

	return total_features, total_coef

def train_and_analyse(_X, _y, features):
	X = _X
	Y = _y
	cv_l = cross_validation.KFold(X.shape[0], n_folds=5,
								shuffle=True, random_state=1)
	ranks = {}

	# lr = LinearRegression(normalize=True)
	# lr.fit(X, Y)
	# ranks["Linear reg"] = rank_to_dict(np.abs(lr.coef_), features)
	

	# ridge = RidgeCV(cv=cv_l)
	# ridge.fit(X, Y)
	# ranks["Ridge"] = rank_to_dict(np.abs(ridge.coef_), features)
	
	# Run the RandomizedLasso: we use a paths going down to .1*alpha_max
    # to avoid exploring the regime in which very noisy variables enter
    # the model
	lasso = LassoCV(cv=cv_l, n_jobs=2, normalize=True, tol=0.0001, max_iter=170000)
	lasso.fit(X, Y)
	ranks["Lasso"] = rank_to_dict(np.abs(lasso.coef_), features)
	
	rlasso = RandomizedLasso(alpha=lasso.alpha_, random_state=42)
	rlasso.fit(X, Y)
	ranks["Stability"] = rank_to_dict(np.abs(rlasso.scores_), features)
	
	# rfe = RFE(lr, n_features_to_select=1)
	# rfe.fit(X,Y)
	# ranks["RFE"] = rank_to_dict(np.array(rfe.ranking_).astype(float), features, order=-1)

	# rf = RandomForestRegressor(n_estimators=400)
	# rf.fit(X,Y)
	# ranks["RF"] = rank_to_dict(rf.feature_importances_, features)

	# f, pval  = f_regression(X, Y, center=True)
	# ranks["Corr."] = rank_to_dict(np.nan_to_num(f), features)

	# mine = MINE()
	# mic_scores = []
	# for i in range(X.shape[1]):
	#    mine.compute_score(X[:,i], Y)
	#    m = mine.mic()
	#    mic_scores.append(m)
	
	# ranks["MIC"] = rank_to_dict(mic_scores, features) 

	# r = {}
	# for name in features:
	#     r[name] = round(np.mean([ranks[method][name] 
	#                              for method in ranks.keys()]), 2)
	 
	# methods = sorted(ranks.keys())
	# ranks["Mean"] = r
	# methods.append("Mean")
	
	ranks = pd.DataFrame(ranks)

	selection_feature = ranks[ranks.Stability > 0.2].index.values
	coef = ranks[ranks.Stability > 0.2].Stability.values
	print(selection_feature.shape, coef.shape)
	return ranks, selection_feature, coef


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

def rank_to_dict(ranks, names, order=1, ratio=1):
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
