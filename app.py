import streamlit as st
import datetime
import pandas as pd
from model_prediction.predict_allocation import predict

today_date = datetime.date.today()
district_file='district_mapping.csv'
SEASONAL_YEAR = today_date.year + 1 if today_date.month >= 6 else today_date.year
y = predict()

st.title(f"Water Allocation Prediction for {SEASONAL_YEAR} summer season")
district_map = pd.read_csv(district_file)
option = st.selectbox(
    'Please select the irrigation district that you want to see',
    ['All'] + district_map['Irrigation District'].to_list()
)

clicked = st.button("Run the model")
if clicked:
    if option != 'All':
        allocation = y[y['District'] == option]['Water Allocation'].values[0]
        st.subheader(f"The estimiated allocation is {"%.2f" % allocation} AF/A")
    else:
        st.dataframe(y, hide_index=True)