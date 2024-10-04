from model_prediction.get_data import get_historical_water_from_all_station
from model_prediction.train_data import map_reservior_to_code, construct_features, read_cdec_data

import pandas as pd
import xgboost as xgb
import datetime


model_filename = 'predict_allocation.model'
current_data_filename = 'current_water_data_all_station.json'
starting_date_of_2025 = '2023-06-01'
today_date = datetime.date.today()
district_file='district_mapping.csv'
station_code_file='station_code.csv'
SEASONAL_YEAR = today_date.year + 1 if today_date.month >= 6 else today_date.year

def load_model():
    model = xgb.XGBRegressor()
    model.load_model(model_filename)  
    return model

def prep_new_data():
    district_map = pd.read_csv(district_file)
    station = pd.read_csv(station_code_file)
    station_mapping = dict(zip(station['Station Name'],station['Station Code']))
    district_map['Res_Code'] = district_map.apply(lambda row: map_reservior_to_code(row['Reservoir'], station_mapping), axis=1)
    district_code_mapping = dict(zip(district_map['Irrigation District'], district_map['Res_Code']))

    current_water_data = read_cdec_data(current_data_filename)
    df_water_temp = pd.DataFrame()
    for k,v in district_code_mapping.items():
        res = v.split(',')
        df_water_temp = pd.concat([df_water_temp, construct_features(k, res, current_water_data)])
    df_water_temp['Year'] = df_water_temp['seasonal_year']
    df_water_temp.drop(columns='seasonal_year', inplace=True)
    predict_data = df_water_temp[df_water_temp['Year'] == SEASONAL_YEAR]
    return predict_data.reset_index(drop=True)

def predict():
    get_historical_water_from_all_station(start_date=starting_date_of_2025, end_date=today_date.strftime('%Y-%m-%d'), filename=current_data_filename)
    model = load_model()
    feature_order= model.get_booster().feature_names
    X_today = prep_new_data()
    X_today['District'] = X_today['District'].astype('category')
    y_predict = model.predict(X_today[feature_order])    
    y = pd.DataFrame(y_predict, columns=['Water Allocation'])
    y['District'] = X_today['District']
    return y[['District','Water Allocation']]