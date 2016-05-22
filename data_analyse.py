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
from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, f_regression
from sklearn.linear_model import (LinearRegression, LassoCV, RidgeCV, RandomizedLasso, lasso_stability_path)
from minepy import MINE

import util as ut


def comprehensive_features_analyse(_df, store_item_nbrs):
	df = _df.copy()
	plt.figure(figsize=(16, 26))
	X, y = ut.get_train_data(df)
	feature_list = X.columns.values[2:]
	total_weight = 0
	total_rank = pd.DataFrame()
	total_features = []
	total = 0
	for sno, ino in store_item_nbrs:
		if(sno == 35):
			continue
		X_1 = X[(X.store_nbr == sno) & (X.item_nbr == ino)]
		X_1 = X_1.drop(['store_nbr', 'item_nbr'], axis=1)
		y_1 = y[X_1.index.values]
		y_1 = y_1.values
		weight = y_1[y_1 > 0].shape[0]
		total_weight += weight
		features = feature_list
		rank, selection_feature = train_and_analyse(X_1, y_1, sno, ino)
		if(len(total_rank) == 0):
			total_rank = rank
		total_features.append(selection_feature)
		total_rank += rank * weight
		total += 1
		print('done', total)
	total_rank /= total_weight
	# total_rank.plot.barh(stacked=False)
	total_rank.to_pickle('Analyse/total_rank_time-specific')
	# plt.show()
	# plt.close()

	return total_features

def train_and_analyse(_X, _y, sno, ino):
	X = _X.copy()
	Y = _y
	features = X.columns.values
	cv_l = cross_validation.KFold(X.shape[0], n_folds=5,
								shuffle=True, random_state=1)
	ranks_linear = {}
	ranks_nonlinear= {}
	ranks_path = {}
	ranks = {}

	selection_feature = []

	time_feature_1 = [
					'date2j'
					]
	time_feature_2 = [
					'day',
					'month',
					'year'
					]

	time_feature_3 = [
					'is_2012', 
					'is_2013', 
					'is_2014',
					'fall', 
					'winter', 
					'spring',
					'summer'
					]

	time_feature_4 = [
					'weekday',
					'is_weekend', 
					'is_holiday', 
					'is_holiday_weekday', 
					'is_holiday_weekend',
					]

	time_feature_5 = [
					'MemorialDay', 
					'MothersDay', 
					'BlackFridayM3',
					'BlackFriday1', 
					'NewYearsDay', 
					'IndependenceDay', 
					'VeteransDay',
					'BlackFriday2', 
					'NewYearsEve', 
					'BlackFriday3', 
					'ChristmasDay',
					'BlackFridayM2', 
					'ThanksgivingDay', 
					'Halloween', 
					'EasterSunday',
					'ChristmasEve', 
					'ValentinesDay', 
					'PresidentsDay', 
					'ColumbusDay',
					'MartinLutherKingDay', 
					'LaborDay', 
					'FathersDay', 
					'BlackFriday'
					]

	weather_feature =  [
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
					'raincode'
					]
	temp = time_feature_1 + time_feature_2 + time_feature_3 + time_feature_4 + time_feature_5
	X_f1 = X[temp].values
	# lr = LinearRegression(normalize=True)
	# lr.fit(X, Y)
	# ranks["Linear reg"] = rank_to_dict(np.abs(lr.coef_), features)
	
	f, pval  = f_regression(ut.get_processed_X_A(X_f1), Y, center=True)
	ranks["F_regr"] = pd.Series(rank_to_dict(np.nan_to_num(f), temp))
	# print('asd')
	# mi = mutual_info_regression(ut.get_processed_X_A(X_f1), Y)
	# mi /= np.max(mi)
	# ranks['MI'] = Pd.Series()

	mine = MINE()
	mic_scores = []
	for i in range(ut.get_processed_X_A(X_f1).shape[1]):
	   mine.compute_score(ut.get_processed_X_A(X_f1)[:,i], Y)
	   m = mine.mic()
	   mic_scores.append(m)
	
	ranks["MIC"] = pd.Series(rank_to_dict(mic_scores, temp))
	


	# ridge.fit(X, Y)
	# ranks["Ridge"] = rank_to_dict(np.abs(ridge.coef_), features)
	
	# Run the RandomizedLasso: we use a paths going down to .1*alpha_max
	# to avoid exploring the regime in which very noisy variables enter
	# the model
	# rlasso = RandomizedLasso(alpha='bic', normalize=True)
	# rlasso.fit(X_f1, Y)
	# ranks_linear["Stability"] = pd.Series(rlasso.scores_)

	# alpha_grid, scores_path = lasso_stability_path(X_f1, Y, random_state=42,
 #                                                   eps=0.00005, n_grid=500)
	# for alpha, score in zip(alpha_grid, scores_path.T):
	# 	ranks_path[alpha] = score
	# ranks_path = pd.DataFrame(ranks_path).transpose()
	# ranks_path.columns = temp
	# plt.figure()
	# ranks_path.plot()
	# plt.show()
	# selection_feature.extend(ranks_linear[ranks_linear.F_regr > 0.1].index.values.tolist())
	# selection_feature.extend(ranks_linear[ranks_linear.MIC > 0.1].index.values.tolist())
	# selection_feature.extend(ranks_linear[ranks_linear.Stability > 0.1].index.values.tolist())
#-------------------------------

	# rf = RandomForestRegressor(n_estimators=150, max_depth=4, n_jobs=4, random_state=1)
	rf = ut.get_regression_model('RandomForest', 0)
	scores = []
	for i in range(X_f1.shape[1]):
	 score = cross_val_score(rf, X_f1[:, i:i+1].astype(float), Y, scoring="r2", cv=ShuffleSplit(len(X_f1), 3, .3), n_jobs=2)
	 scores.append(round(np.mean(score), 3))

	ranks['RF'] = pd.Series(rank_to_dict(np.abs(scores), temp)) 

	ranks = pd.DataFrame(ranks)
	print(ranks)
	selection_feature.extend(ranks[ranks.RF > 0.1].index.values.tolist())
	selection_feature.extend(ranks[ranks.MIC >= 0.1].index.values.tolist())
	selection_feature.extend(ranks[ranks.F_regr >= 0.1].index.values.tolist())
#-------------------------------
	selection_feature = list(set(selection_feature))
	print(selection_feature)
	# ridge = RidgeCV(cv=cv_l)
	# rfe = RFE(ridge, n_features_to_select=1)
	# rfe.fit(X[selection_feature],Y)
	# ranks["RFE"] = pd.Series(rank_to_dict(np.array(rfe.ranking_).astype(float), selection_feature, order=1))
	# ranks = pd.DataFrame(ranks)
	# print(ranks)
	# r = {}
	# for name in features:
	#     r[name] = round(np.mean([ranks[method][name] 
	#                              for method in ranks.keys()]), 2)
	 
	# methods = sorted(ranks.keys())
	# ranks["Mean"] = r
	# methods.append("Mean")

	path = 'Analyse/store_{}/'.format(sno)
	mkdir_p(path)
	path += 'item_{}_(pair_analyse)'.format(ino)
	ranks.to_pickle(path)

	path += '.png'
	p.clf()
	p.cla()
	plt.figure(figsize=(16, 26))
	ranks.plot.barh(stacked=True)
	p.savefig(path, bbox_inches='tight', dpi=300)
	plt.close()

	return ranks, selection_feature


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
	if np.mean(ranks) == 0:
		ranks+=1
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
