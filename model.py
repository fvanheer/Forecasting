#####################################################################################################################################
### FORECAST COMMODITY PRICES ###
#####################################################################################################################################
import pandas as pd
import glob2 as glob
import numpy as np
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.plot import plot_plotly
import plotly.express as px
import plotly.offline as py
import datetime as dt

#####################################################################################################################################
### IMPORT DATA ###
#####################################################################################################################################
df = pd.read_csv('temp_data.csv')

#convert string to date format
df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
df['Daily minimum temperatures'] = df['Daily minimum temperatures'].map(lambda x: x.lstrip('?'))
df['Daily minimum temperatures'] = df['Daily minimum temperatures'].astype(float)

#####################################################################################################################################
### REVIEW THE TIMESERIES DATA ###
#####################################################################################################################################
#plot the yield of US Soybean crop
fig = px.line(df, x='Date', y='Daily minimum temperatures')
fig.show()

#####################################################################################################################################
### Train the Model ###
#####################################################################################################################################

#format the data to required format for the library {Date: 'DS', Variable: 'Y'}
df = df.rename(columns={'Date':'ds', 'Daily minimum temperatures': 'y'})
forecast_training = df[['ds','y']]
forecast_training['y'] = np.where(
    forecast_training['y'] == 0, 0.1, forecast_training['y']
)

#fit model
m = Prophet()
m.fit(forecast_training)

#####################################################################################################################################
### Prediction incl. Parameters ###
#####################################################################################################################################

future = m.make_future_dataframe(periods=365)
future.tail() #tail end of the data

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

#####################################################################################################################################
### REVIEW MODEL PERFORMANCE ###
#####################################################################################################################################

#plot the forecast
fig1 = m.plot(forecast)

#plot decomposition of seasonality and trends
fig2 = m.plot_components(forecast)

#Interactive plots
py.init_notebook_mode()

fig = plot_plotly(m, forecast)  # This returns a plotly Figure - interactive plot - use the date slider at the bottom
py.iplot(fig)

#####################################################################################################################################
### RUN DIAGNOSTICS ###
#####################################################################################################################################

#run cross validation
from fbprophet.diagnostics import cross_validation
df_cv = cross_validation(m, initial='720 days', period='180 days', horizon = '365 days')
df_cv.head()

#plot cross validation absolute percent error for forecast
from fbprophet.plot import plot_cross_validation_metric
fig = plot_cross_validation_metric(df_cv, metric='mape')

#####################################################################################################################################
### ADJUST & RE-FORECAST ###
#####################################################################################################################################

#handle outliers




#####################################################################################################################################
###  ###
#####################################################################################################################################

