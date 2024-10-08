import streamlit as st
import datetime
import pandas as pd
from model_prediction.predict_allocation import predict

today_date = datetime.date.today()
district_file='district_mapping.csv'
SEASONAL_YEAR = today_date.year + 1 if today_date.month >= 6 else today_date.year
y, score = predict()

st.title(f"Water Allocation Prediction for {SEASONAL_YEAR} summer season")
st.text(f"The prediction is using the data from {today_date}")

clicked = st.button("Run the model")
if clicked:
    st.dataframe(y, hide_index=True)
    st.text(f'Current confidence level is {(score*100):.2f}%')