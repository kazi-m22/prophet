from pandas import read_csv
import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib.pyplot as plt


df = pd.read_csv('1.csv')
df['y'] = np.log(df['y'])
df.head()

m = Prophet()
m.fit(df)


future = m.make_future_dataframe(periods=365)
future.tail()

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


m.plot(forecast)
plt.show()