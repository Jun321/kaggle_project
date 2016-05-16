import numpy as np 
import pandas as pd 

def submission_to_file(path, RES, store_item_nbrs):
	df_test = pd.read_csv('data/test.csv')
	df_test['log1p'] = 0
	for sno, ino in store_item_nbrs:
		if(sno == 35):
			continue
		df_temp = df_test[(df_test.store_nbr == sno) & (df_test.item_nbr == ino) & (df_test.date != '2013-12-25')]
		index_val = df_temp.index.values
		df_pre = RES[(RES.store_nbr == sno) & (RES.item_nbr == ino)]
		pre_val = df_pre['log1p'].values
		df_test.loc[index_val, 'log1p'] = pre_val

		

	df_test['id'] = df_test['store_nbr'].astype(str) + '_' + df_test['item_nbr'].astype(str) + '_' + df_test['date']
	df_test['units'] = np.exp(df_test['log1p']) - 1
	df_test['units'] = np.maximum(df_test['units'], 0.)
	df_test = df_test[['id', 'units']]
	df_test.to_csv(path, index=False)

