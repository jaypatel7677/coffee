import streamlit as st
import pandas as pd
from prophet import Prophet

st.title("☕ Coffee Demand Forecaster")

# Generate synthetic data
dates = pd.date_range(start="2023-01-01", periods=365)
df = pd.DataFrame({
    "ds": dates,
    "y": 100 + 30*(dates.dayofweek >= 5)  # 100 daily cups + weekend boost
})

# Train and predict
model = Prophet()
model.fit(df)
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

# Show results
st.write("Next 90 days forecast:")
st.line_chart(forecast.set_index("ds")[["yhat"]])
