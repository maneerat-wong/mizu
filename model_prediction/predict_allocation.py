from model_prediction.get_data import get_water_data_from_all_station, get_swe_data
from model_prediction.train_data import map_reservior_to_code, construct_features_daily, read_cdec_data, construct_data_for_daily_model, train_model

import pandas as pd
import xgboost as xgb
import datetime

model_filename = 'predict_allocation.model'
current_data_filename = 'current_water_data_all_station.json'
current_swe_filename = 'current_swe_data_all_station.json'
starting_date_of_2025 = '2023-06-01'
today_date = datetime.date.today()
district_file='district_mapping.csv'
station_code_file='station_code.csv'
swe_station_file='swe_stations.csv'

SEASONAL_YEAR = today_date.year + 1 if today_date.month >= 6 else today_date.year

def load_model():
    model = xgb.XGBRegressor()
    model.load_model(model_filename)  
    return model

def prep_new_data(selected_date, selected_month):
    district_map = pd.read_csv(district_file)
    station = pd.read_csv(station_code_file)
    station_mapping = dict(zip(station['Station Name'],station['Station Code']))
    district_map['Res_Code'] = district_map.apply(lambda row: map_reservior_to_code(row['Reservoir'], station_mapping), axis=1)
    district_code_mapping = dict(zip(district_map['Irrigation District'], district_map['Res_Code']))
    swe_station = pd.read_csv(swe_station_file)

    swe_all_df = pd.DataFrame()
    for region in ['CS','NS','SS']:
        swe_df = read_cdec_data(f'{region}_{current_swe_filename}', cutting_off_month=6)
        swe_df['SWE Region'] = region
        swe_all_df = pd.concat([swe_all_df, swe_df])

    selected_swe_df = swe_all_df[((swe_all_df.date.dt.month == selected_month) & (swe_all_df.date.dt.day == selected_date))]

    df_swe_temp = pd.DataFrame()
    for _, row in district_map.iterrows():
        regions = [region.strip() for region in row['SWE Region'].strip().split(',')]
        all_related_swe_stations = swe_station[swe_station['SWE Region'].isin(regions)]['Stations'].values
        temp = []
        for l in all_related_swe_stations:
            temp += l.split(',')
        df_swe_temp = pd.concat([df_swe_temp, construct_features_daily(row['Irrigation District'], temp, selected_swe_df)])

    df_swe_temp['Year'] = df_swe_temp['seasonal_year']
    df_swe_temp.drop(columns='seasonal_year', inplace=True)
    df_swe_temp['temp'] = df_swe_temp['District'] + '_' + df_swe_temp['Year'].astype(str)

    current_water_data = read_cdec_data(current_data_filename)
    selected_water_df = current_water_data[((current_water_data.date.dt.month == selected_month) & (current_water_data.date.dt.day == selected_date))]
    
    df_water_temp = pd.DataFrame()
    for k,v in district_code_mapping.items():
        res = v.split(',')
        df_water_temp = pd.concat([df_water_temp, construct_features_daily(k, res, selected_water_df)])
    df_water_temp['Year'] = df_water_temp['seasonal_year']
    predict_data = df_water_temp[df_water_temp['Year'] == SEASONAL_YEAR]

    #Combine SWE data with water data
    predict_data = predict_data.merge(df_swe_temp[df_swe_temp['Year'] == SEASONAL_YEAR], on='District', how='left', suffixes=('','_swe'))
    predict_data.drop(columns=['Year_swe','temp','seasonal_year'], inplace=True)
    
    return predict_data.reset_index(drop=True)

def predict():
    selected_date = today_date.day
    selected_month = today_date.month

    train_df = construct_data_for_daily_model(selected_date, selected_month)
    model, score = train_model(train_df)
    feature_order= model.get_booster().feature_names

    get_water_data_from_all_station(start_date=starting_date_of_2025, end_date=today_date.strftime('%Y-%m-%d'), filename=current_data_filename)
    get_swe_data(start_date=starting_date_of_2025, end_date=today_date.strftime('%Y-%m-%d'), filename=current_swe_filename)
    
    X_today = prep_new_data(selected_date, selected_month)
    X_today['District'] = X_today['District'].astype('category')
    y_predict = model.predict(X_today[feature_order])    
    y = pd.DataFrame(y_predict, columns=['Water Allocation'])
    y['District'] = X_today['District']
    return y[['District','Water Allocation']], score