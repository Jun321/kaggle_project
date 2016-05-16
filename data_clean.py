import numpy as np
import pandas as pd

def get_precip(precip):
    if(precip == 'M' or precip == '  T'):
        return 0
    else:
        return np.float(precip)

def get_temp(temp):
    if(temp == 'M' ):
        return 60
    else:
        return np.float(temp)

def weather_process(_df):
    weather_df = _df.copy()
    weather_df['date2'] = pd.to_datetime(weather_df.date)
    weather_df['high_precip'] = 0
    weather_df['preciptotal'] = weather_df['preciptotal'].map(lambda x: get_precip(x))
    weather_df['high_precip'] = weather_df['preciptotal'].map(lambda x: 1 if (x >= 0.75) else 0)
    weather_df['snowfall'] = weather_df['snowfall'].map(lambda x: get_precip(x))
    weather_df['high_snow'] = weather_df['snowfall'].map(lambda x: 1 if x >= 1.5 else 0)
    weather_df['avgspeed'] = weather_df['avgspeed'].map(lambda x: get_precip(x))
    weather_df['windy'] = weather_df['avgspeed'].map(lambda x: 1 if x >= 18 else 0)
    weather_df['temp_missing'] = weather_df['tavg'].map(lambda x: 1 if x == 'M' else 0)
    weather_df['tavg'] = weather_df['tavg'].map(lambda x: get_temp(x))
    weather_df['hot'] = weather_df['tavg'].map(lambda x: 1 if x >= 80 else 0)
    weather_df['cold'] = weather_df['tavg'].map(lambda x: 1 if x <= 32 else 0)
    weather_df['frigid'] = weather_df['tavg'].map(lambda x: 1 if x <= 15 else 0)
    weather_df['thunder'] = weather_df['codesum'].map(lambda x: 1 if 'TS' in x else 0)
    weather_df['snowcode'] = weather_df['codesum'].map(lambda x: 1 if 'SN' in x else 0)
    weather_df['raincode'] = weather_df['codesum'].map(lambda x: 1 if 'RA' in x else 0)
    weather_df = weather_df[['high_precip', 'preciptotal', 'snowfall',
                            'high_snow', 'avgspeed', 'windy', 
                            'temp_missing', 'tavg', 'hot',
                            'cold', 'frigid', 'thunder',
                            'snowcode', 'raincode', 'date2',
                            'station_nbr']]
    return weather_df

def create_vaild_item_store_combinations(_df):
    df = _df.copy()
    df['log1p'] = np.log(df['units'] + 1)

    g = df.groupby(["store_nbr", "item_nbr"])['log1p'].mean()
    g = g[g > 0.0]

    store_nbrs = g.index.get_level_values(0)
    item_nbrs = g.index.get_level_values(1)

    store_item_nbrs = sorted(zip(store_nbrs, item_nbrs), key=lambda t: t[1] * 10000 + t[0])
    # store_item_nbrs = pd.DataFrame(store_item_nbrs, columns=['store_nbrs', 'item_nbrs'])
    # store_item_nbrs.to_pickle('model/store_item_nbrs.pkl')
    return store_item_nbrs


def get_filted_data(_df, store_item_nbrs, is_train):
    df_train = _df
    df1 = pd.DataFrame()
    for sno, ino in store_item_nbrs:
        df = df_train[(df_train.store_nbr == sno) & (df_train.item_nbr == ino)]
        df1 = df1.append(df)
    df1.reset_index(drop=True, inplace=True)

    exclude_date = pd.to_datetime("2013-12-25")
    df1['date2'] = pd.to_datetime(df1['date'])
    if is_train:
        df1['log1p'] = np.log(df1['units'] + 1)
    df1 = df1[df1.date2 != exclude_date]
    return df1
