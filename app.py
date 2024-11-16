# This is the main file that is used to produce the frontend on the streamlit

import streamlit as st
import datetime
import pytz
from model_prediction.predict_allocation import train_daily_model, predict

eastern_now = datetime.datetime.now(pytz.timezone('US/Eastern'))

data_date = eastern_now.date() - datetime.timedelta(days=1)
SEASONAL_YEAR = data_date.year + 1 if data_date.month >= 6 else data_date.year

st.title(f"Water Allocation Prediction for {SEASONAL_YEAR} summer season")
st.text(f'Today is {eastern_now.date()}')
clicked = st.button("Run the model")
if clicked:
    with st.spinner(f'Running the model for {data_date}... This may take a while'):
        model, _, _, _ = train_daily_model(data_date)
        y = predict(model, data_date)
    st.dataframe(y, hide_index=True)
    st.text(f"The prediction is using the data from end of {data_date}")
    