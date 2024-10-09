import streamlit as st
import datetime
import time
from model_prediction.predict_allocation import train_daily_model, predict

today_date = datetime.date.today()
SEASONAL_YEAR = today_date.year + 1 if today_date.month >= 6 else today_date.year

st.title(f"Water Allocation Prediction for {SEASONAL_YEAR} summer season")

clicked = st.button("Run the model")
if clicked:
    with st.spinner('Running the model'):
        model, score = train_daily_model()
        y = predict(model)
    st.text(f"The prediction is using the data from {today_date}")
    st.dataframe(y, hide_index=True)
    st.text(f'Current confidence level is {(score*100):.2f}%')