import numpy as np
import pandas as pd

def get_holidays(fpath):
    # holidays are from http://www.timeanddate.com/holidays/us/ , holidays and some observances
    
    f = open(fpath)
    lines = f.readlines()
    lines = [line.split(" ")[:3] for line in lines]
    lines = ["{} {} {}".format(line[0], line[1], line[2]) for line in lines]
    lines = pd.to_datetime(lines)
    return pd.DataFrame({"date2":lines})

def get_holiday_names(fpath):
    # holiday_names are holidays + around Black Fridays
    
    f = open(fpath)
    lines = f.readlines()
    lines = [line.strip().split(" ")[:4] for line in lines]
    lines_dt = ["{} {} {}".format(line[0], line[1], line[2]) for line in lines]
    lines_dt = pd.to_datetime(lines_dt)
    lines_hol = [line[3] for line in lines]
    return pd.DataFrame({"date2":lines_dt, "holiday_name":lines_hol})

def get_preprocessed_data(_df, weather_df, key_df):
	df = _df.copy()

	df['date2j'] = (df.date2 - pd.to_datetime("2012-01-01")).dt.days
	df['weekday'] = df.date2.dt.weekday + 1
	df['day'] = df.date2.dt.day
	df['month'] = df.date2.dt.month
	df['year'] = df.date2.dt.year
	df['is_2012'] = df['year'].map(lambda x: 1 if (x == 2012) else 0)
	df['is_2013'] = df['year'].map(lambda x: 1 if (x == 2013) else 0)
	df['is_2014'] = df['year'].map(lambda x: 1 if (x == 2014) else 0)
	df['is_weekend'] = df.date2.dt.weekday.isin([5,6])
	df['is_holiday'] = df.date2.isin(holidays.date2)
	df['is_holiday_weekday'] = df.is_holiday & (df.is_weekend == False)
	df['is_holiday_weekend'] = df.is_holiday & df.is_weekend
	df = pd.merge(df, holiday_names, on='date2', how = 'left')
	df.loc[df.holiday_name.isnull(), "holiday_name"] = ""

	for holiday in set(holiday_names.holiday_name.values):
		if(holiday != ""):
			df[holiday] = df['holiday_name'] == holiday
			df[holiday] = np.where(df[holiday], 1, 0)
	df.is_weekend = np.where(df.is_weekend, 1, 0)
	df.is_holiday = np.where(df.is_holiday, 1, 0)
	df.is_holiday_weekday = np.where(df.is_holiday_weekday, 1, 0)
	df.is_holiday_weekend = np.where(df.is_holiday_weekend, 1, 0)

	df = pd.merge(df, key_df, on='store_nbr')
	df = pd.merge(df, weather_df, on=['date2', 'station_nbr'])
	df['fall'] = df['month'].map(lambda x: 1 if (9 <= x < 12) else 0)
	df['winter'] = df['month'].map(lambda x: 1 if (x==12 or x==1 or x==2) else 0)
	df['spring'] = df['month'].map(lambda x: 1 if (3 <= x < 6) else 0)
	df['summer'] = df['month'].map(lambda x: 1 if (6 <= x < 9) else 0)

	df = df[df['store_nbr'] != 35]
	df = df.reset_index(drop = True)

	return df

holidays = get_holidays("holidays.txt")
holiday_names = get_holiday_names("holiday_names.txt")
