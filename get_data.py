import requests
from bs4 import BeautifulSoup
import json
import pandas as pd

class MyDict(dict):
    def __missing__(self, key):
        return key

#site to get updated data
CDEC_API_URL = "https://cdec.water.ca.gov/dynamicapp/req/JSONDataServlet"
SWE_URL = "https://cdec.water.ca.gov/snowapp/sweqdate.action"


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
def get_historical_water_from_all_station(dur_code='D', station_code_file='station_code.csv', start_date='1988-6-1', end_date='2024-05-30', filename = 'all_station_historical_water.json'):
    station = pd.read_csv(station_code_file)
    all_water_data = []
    for code in station['Station Code']:
        all_water_data += get_json_from_cdec(code, start_date, end_date, dur_code=dur_code)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(all_water_data, f, ensure_ascii=False)


#This specific data doesn't have the data before 2001
def get_swe_data(query_date):
    """Get the Snow Water Equivalent for three regions (North, Central, South)

    Args:
        query_date (String): query date in the format of DD-MM-YYYY but MM in Month name e.g. Jun

    Returns:
        SWE for three regions in json 
    """

    swe_data = {"date":query_date}

    payload = {'querydate':query_date}
    response = requests.request("GET", SWE_URL, params=payload)
    webpage = response.text
    soup = BeautifulSoup(webpage, features="lxml")
    mydivs = soup.findAll("div", {"class":"block_with_rounded_corners"})
    for div in mydivs:
        region = div.find("h3")
        trs = div.findAll("tr")
        for tr in trs:
            tds = tr.findAll("td")
            key_value = [td.text for td in tds]
            if 'snow water equivalent' in key_value[0]: 
                swe_data[region.text] = key_value[1]

    return swe_data


def get_most_recent_swe(json_filename='swe_from_1990.json'):
    data = []
    for d in pd.date_range(datetime.date(2000,1,1), datetime.date.today()):
        data.append(get_swe_data(d.strftime('%d-%b-%Y')))
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    


