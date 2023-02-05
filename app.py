# pip install streamlit fbprophet yfinance plotly

# !pip install yfinance
# !pip install prophet

import streamlit as st
from datetime import date, datetime, timedelta

import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# TODAY = date.today().strftime("%Y-%m-%d")
TODAY = datetime.today()

st.title('Stock Forecast')

st.markdown('This app is built to predict the stock market performance')

stocks = ('TSLA', 'FB', 'NVDA', 'BABA', 'GOOG', 'AAPL', 'MSFT', 'GME', 'AMZN', 'XIACF')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

new_resolution = st.radio(
	"Do you want to get the higher resolution or short time interval, please choose one:",
	('In 1 day', 'In 1 hour', 'In 5 minutes'))

if new_resolution == 'In 5 minutes':
	new_interval = "5m"
	START = TODAY - timedelta(days=30)
elif new_resolution == 'In 1 hour':
	new_interval = "1h"
	START = TODAY - timedelta(days=365)
else:	
	new_interval = "1d"
	START = "2018-01-01"

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY, interval = new_interval)
    data.reset_index(inplace=True)
    return data

	
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('... Data loaded, well done!')

@st.cache
def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv().encode('utf-8')

csv = convert_df(data)

st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='stock_data.csv',
    mime='text/csv',
 )


st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())
    
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)

