import requests
import json
import pandas as pd
import datetime

#site to get updated data
CDEC_API_URL = "https://cdec.water.ca.gov/dynamicapp/req/JSONDataServlet"
SWE_URL = "https://cdec.water.ca.gov/snowapp/sweqdate.action"

TRAINING_DATA_START_DATE='1988-6-1'
TRAINING_DATA_END_DATE='2024-05-30'

def get_json_from_cdec(station_code, start_date, end_date, dur_code, sensor_num = 15):
    """Get the water data in AF for each reservoir from cdec website

    Args:
        station_code (string): 3 character station code
        start_date (String): start date to get the data in YYYY-MM-DD format
        end_date (String): last date to get the data in YYYY-MM-DD format
        sensor_num (int): specific sensor number, 15 is water storage, please refer to https://cdec.water.ca.gov/dynamicapp/wsSensorData
        dur_code (String): either D(daily), W(weekly), or M(monthly)

    Returns:
        json: water data from start_date to end_date
    """
    payload = {'Stations' : station_code,
               'SensorNums' : sensor_num,
               'dur_code' : dur_code,
               'Start': start_date,
               'End' : end_date}
    
    response = requests.request("GET", CDEC_API_URL, params=payload)
    return response.json()


#Gat Training data from all station in station_code.csv
def get_water_data_from_all_station(start_date=TRAINING_DATA_START_DATE, end_date=TRAINING_DATA_END_DATE, dur_code='D', station_code_file='station_code.csv', filename = 'all_station_historical_water.json'):
    station = pd.read_csv(station_code_file)
    all_water_data = []
    for code in station['Station Code']:
        all_water_data += get_json_from_cdec(code, start_date, end_date, dur_code=dur_code)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(all_water_data, f, ensure_ascii=False)


def get_swe_data(start_date=TRAINING_DATA_START_DATE, end_date=TRAINING_DATA_END_DATE, dur_code='D', sensor_num=3, station_code_file='swe_stations.csv', filename ='training_data_swe.json'):
    """Get the Snow Water Equivalent for three regions (North, Central, South)

    Args:
        sensor_num (int): sensor number for snow water content
    """
    swe_stations = pd.read_csv(station_code_file)
    for _, row in swe_stations.iterrows():
        swe_json = get_json_from_cdec(row['Stations'], start_date, end_date, dur_code, sensor_num)
        with open(f'{row['SWE Region']}_{filename}', 'w', encoding='utf-8') as f:
            json.dump(swe_json, f, ensure_ascii=False)



