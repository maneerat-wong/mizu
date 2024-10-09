import streamlit as st
import datetime
from pytz import timezone
import time
from model_prediction.predict_allocation import train_daily_model, predict

data_date = datetime.date.today() - datetime.timedelta(days=1)
SEASONAL_YEAR = data_date.year + 1 if data_date.month >= 6 else data_date.year

st.title(f"Water Allocation Prediction for {SEASONAL_YEAR} summer season")
st.text(f'Today is {datetime.date.today()}')
clicked = st.button("Run the model")
if clicked:
    with st.spinner(f'Running the model for {data_date}... This may take a while'):
        model, score = train_daily_model()
        y = predict(model)
    st.text(f"The prediction is using the data from end of {data_date} and the confidence level is {(score*100):.2f}%")
    st.dataframe(y, hide_index=True)