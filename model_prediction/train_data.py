import pandas as pd
import json
import numpy as np
import datetime
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

MODEL_FILENAME = 'predict_allocation.model'
HISTORICAL_DAILY_WATER_FILE = 'all_station_historical_water.json'
HISTORICAL_MONTHLY_WATER_FILE = 'all_station_historical_water_M.json'
HITORICAL_SWE_FILE = 'training_data_swe.json'

def read_cdec_data(train_file, cutting_off_month=6):
    """Read CDEC json file and clean the dataframe

    Args:
        train_file (str): name of the file that we downloaded from cdec website
        cutting_off_month (int, optional): This is used to set the seasonal year. Defaults to 6.

    Returns:
        DataFrmae: dataframe of the data from CDEC with minimal cleaning
    """
    with open(train_file, 'r') as file:
        cdec_data = json.load(file)
    df = pd.json_normalize(cdec_data)
    df['date'] = pd.to_datetime(df['date'])

    # -9999 appear only 0.001 percent and those rows are unavailable so I will remove these data out
    df = df[df['value'] != -9999]
    df['seasonal_year'] = np.where(df['date'].dt.month >= cutting_off_month, df['date'].dt.year + 1, df['date'].dt.year)
    return df
    

def map_reservoir_to_code(reservoir_list, station_mapping):
    """translate the reservoir name for each irrigation district with the code on CDEC website

    Args:
        reservoir_list (list): list of the reservoir for a specific district
        station_mapping (dict): mapping between the reservoir name and the reservoir code based on CDEC website

    Returns:
        str: list of the reservoir code 
    """
    res_code = []
    for res in reservoir_list.split(','):
        res_code.append(station_mapping[res.strip()])
    return ",".join(res_code)


def flatten_multiindex(multi_index_df):
    """Flatten the multi index dataframe to single index

    Args:
        multi_index_df (DataFrame): Dataframe that contains multi index

    Returns:
        DataFrame: Flattened multi-index dataframe to the single index dataframe
    """
    flat_cols = []
    for col in multi_index_df.columns:
        if col[1] == '':
            flat_cols.append(col[0])
        else:
            flat_cols.append(col[0] + '_' + col[1])
    multi_index_df.columns = flat_cols
    return multi_index_df


def construct_features_daily(district, resorvoir_list, selected_date_df):
    """Add features to dataframe for the training/prediction

    Args:
        district (str): name of the irrigation district
        resorvoir_list (list): list of the resorvoir 
        selected_date_df (DataFrame): dataframe which contains only the data from a specific date

    Returns:
        DataFrame: Aggregated dataframe
    """
    selected_df = selected_date_df[selected_date_df['stationId'].isin(resorvoir_list)]
    yearly = selected_df.groupby('seasonal_year',as_index=False).agg({'value':['mean','min','max'],
                                                                      'stationId':'count'})
    yearly = flatten_multiindex(yearly)
    yearly = yearly.rename(columns={'stationId':'no_stations'})
    yearly['change_in_a_year_mean'] = yearly['value_mean'].diff().fillna(0)
    yearly['change_in_a_year_min'] = yearly['value_min'].diff().fillna(0)
    yearly['change_in_a_year_max'] = yearly['value_max'].diff().fillna(0)
    yearly['District'] = district
    return yearly


#Known issue : CFW Daily info doesn't have the data before 2021, therefore, will use the monthly data for the missing one to train the model
def construct_data_for_daily_model(selected_date, selected_month, district_file='district_mapping.csv', station_code_file='station_code.csv', allocation_file='Allocation_data.csv', swe_station_file='swe_stations.csv'):

    """Construct the dataframe in order to train the model

    Returns:
        DataFrame: dataframe which is ready to be used for training
    """
    district_map = pd.read_csv(district_file)
    station = pd.read_csv(station_code_file)
    swe_station = pd.read_csv(swe_station_file)

    # This part is for SWE data
    swe_all_df = pd.DataFrame()
    for region in ['CS','NS','SS']:
        swe_df = read_cdec_data(f'{region}_{HITORICAL_SWE_FILE}', cutting_off_month=6)
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
    
    # This part is for water storage data
    station_mapping = dict(zip(station['Station Name'],station['Station Code']))
    district_map['Res_Code'] = district_map.apply(lambda row: map_reservoir_to_code(row['Reservoir'], station_mapping), axis=1)

    label_data = pd.read_csv(allocation_file)
    district_code_mapping = dict(zip(district_map['Irrigation District'], district_map['Res_Code']))
    all_district = list(district_code_mapping)
    train_data = []

    for district in all_district:
        for year in label_data['Year']:
            train_data.append([district, year])
    train_df = pd.DataFrame(train_data, columns=['District','Year'])
   
    # only use for mapping allocation
    train_df['temp'] = train_df['District'] + '_' + train_df['Year'].astype(str)
    all_allocation_mapping = {}
    for district in all_district:
        label_data['temp'] = district + '_' + label_data['Year'].astype(str)
        all_allocation_mapping.update(label_data[['temp',district]].set_index('temp').to_dict()[district])
    train_df['allocation'] = train_df['temp'].map(all_allocation_mapping)

    df_water_daily = read_cdec_data(HISTORICAL_DAILY_WATER_FILE)

    # Add the info for the missing year for daily CFW, DRE, MCS, MDO
    df_water_monthly = read_cdec_data(HISTORICAL_MONTHLY_WATER_FILE) # Will use this for CFW data < 2022 only
    cfw_monthly = df_water_monthly[(df_water_monthly['date'].dt.date < datetime.date(2021,9,1))]
    if selected_date != 1:
        cfw_monthly['date'] = cfw_monthly['date'].apply(lambda x: x.replace(day=selected_date))
    
    all_train_data = pd.concat([df_water_daily, cfw_monthly])

    selected_date_df = all_train_data[((all_train_data.date.dt.month == selected_month) & (all_train_data.date.dt.day == selected_date))]
    df_water_temp = pd.DataFrame()
    for k,v in district_code_mapping.items():
        res = v.split(',')
        df_water_temp = pd.concat([df_water_temp, construct_features_daily(k, res, selected_date_df)])

    df_water_temp['Year'] = df_water_temp['seasonal_year']
    df_water_temp.drop(columns='seasonal_year', inplace=True)
    df_water_temp['temp'] = df_water_temp['District'] + '_' + df_water_temp['Year'].astype(str)

    # water data
    train_df = train_df.merge(df_water_temp, on='temp', how='left', suffixes=('','_'))
    train_df.drop(train_df.filter(regex='_$').columns, axis=1, inplace=True)

    # Combine SWE data with water data
    train_df = train_df.merge(df_swe_temp, on='temp', how='left', suffixes=('','_swe'))
    train_df.drop(columns=['District_swe','Year_swe'], inplace=True)

    train_df.drop(columns=['temp'], inplace=True)
    train_df = train_df.dropna(subset=['allocation'])
    return train_df

def cal_error_score(y, y_predict):
    """Calculate the specific error score. The R-square is not a very good indication for the accuracy if the data is concentrated to only 1 point.

    Args:
        y (list): true label
        y_predict (list): predicted label

    Returns:
        float: the error score 
    """
    y_temp = y.reset_index()
    y_temp = y_temp[y_temp['allocation'] > 0]['allocation']
    y_predict = y_predict[y_temp.index]
    return (sum((((y_temp - y_predict)/y_temp) ** 2)) / len(y_temp)) ** 1/2


def train_model(train_df):
    """Train the model to get the model for the prediction

    Args:
        train_df (DataFrame): Dataframe with features for the training

    Returns:
        model and scores
    """
    y = train_df['allocation']
    X = train_df.drop(columns='allocation')
    X['District'] = X['District'].astype('category')
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=65)
    
    ### This part is for model tuning but it takes a while on CPU so I will omit this for the time being

    # params = {
    #     'min_child_weight': [1, 5, 10],
    #     'gamma': [0.5, 1, 1.5, 2, 5],
    #     'subsample': [0.6, 0.8, 1.0],
    #     'colsample_bytree': [0.6, 0.8, 1.0],
    #     'max_depth': [3, 4, 5],
    #     "learning_rate": [0.01, 0.05, 0.10]
    #     }
    #folds = KFold(n_splits = 5, shuffle = True, random_state = 100)
    #random_search = RandomizedSearchCV(xgb.XGBRegressor(enable_categorical='True'), param_distributions=params, scoring='r2', cv=folds)
    #random_search.fit(train_X, train_y)
    #model = xgb.XGBRegressor(enable_categorical='True', **random_search.best_params_)


    model = xgb.XGBRegressor(enable_categorical='True')
    model.fit(train_X, train_y)
    #scores = cross_val_score(model, train_X, train_y, scoring='r2', cv=folds) 
    y_predict = model.predict(test_X)
    r2 = r2_score(test_y, y_predict)
    error_score = cal_error_score(test_y, y_predict)
    mse = mean_squared_error(test_y, y_predict)
    return model, r2, error_score, mse




######## This is only used for experimental/exploration only ####

# def train_model_district(train_df):
#     y = train_df['allocation']
#     X = train_df.drop(columns=['allocation','District'])
#     train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=65)

#     params = {
#         'min_child_weight': [1, 5, 10],
#         'gamma': [0.5, 1, 1.5, 2, 5],
#         'subsample': [0.6, 0.8, 1.0],
#         'colsample_bytree': [0.6, 0.8, 1.0],
#         'max_depth': [3, 4, 5],
#         "learning_rate": [0.01, 0.05, 0.10]
#         }
    
#     #This part is for model tuning but it will take a while so I will comment this out for the time being
#     #folds = KFold(n_splits = 5, shuffle = True, random_state = 100)
#     #random_search = RandomizedSearchCV(xgb.XGBRegressor(enable_categorical='True'), param_distributions=params, scoring='r2', cv=folds)
#     #random_search.fit(train_X, train_y)

#     model = xgb.XGBRegressor(enable_categorical='True')
#     #model = xgb.XGBRegressor(enable_categorical='True', **random_search.best_params_)
#     #model = DecisionTreeRegressor(max_depth=2)
#     model.fit(train_X, train_y)
#     #scores = cross_val_score(model, train_X, train_y, scoring='r2', cv=folds) 
#     y_predict = model.predict(test_X)
#     r2 = r2_score(test_y, y_predict)
#     error_score = cal_error_score(test_y, y_predict)
#     mse = mean_squared_error(test_y, y_predict)
#     test_score = r2
#     #model.save_model(MODEL_FILENAME)
#     return model, test_score, error_score, mse
    
