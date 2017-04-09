from pandas import read_csv
import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib.pyplot as plt


df = pd.read_csv('1.csv')
df['y'] = np.log(df['y'])
df['cap'] = 8.5
m = Prophet(growth='logistic')
m.fit(df)
future = m.make_future_dataframe(periods=1826)
future['cap'] = 8.5
# fcst = m.predict(future)
# m.plot(fcst)
# plt.show()
# m = Prophet(changepoint_prior_scale=0.5)
# forecast = m.fit(df).predict(future)
# m.plot(forecast)
# plt.show()

# m = Prophet(changepoint_prior_scale=0.001)
# forecast = m.fit(df).predict(future)
# m.plot(forecast)
# plt.show()

m = Prophet(changepoints=['2014-01-01'])
forecast = m.fit(df).predict(future)
m.plot(forecast)
plt.show()
