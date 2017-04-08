from pandas import read_csv
import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib.pyplot as plt


df = pd.read_csv('THU_U_day.csv')
df = pd.read_csv('1.csv')
# df['y'] = np.log(df['y'])
df.head()

m = Prophet()
m.fit(df)


future = m.make_future_dataframe(periods=365)
future.tail()

forecast = m.predict(future)
forecast[['Year','MonthOfYear','DayOfMonth','DayOfYear','DayOfCentury','AirPressure(hPa)','AirTemperature(C)','AirTemperatureHygroClip(C)','RelativeHumidity_wrtWater(%)','RelativeHumidity(%)','WindSpeed(m/s)','WindDirection(d)','ShortwaveRadiationDown(W/m2)','ShortwaveRadiationDown_Cor(W/m2)','ShortwaveRadiationUp(W/m2)','ShortwaveRadiationUp_Cor(W/m2)','Albedo_theta<70d','LongwaveRadiationDown(W/m2)','LongwaveRadiationUp(W/m2)','CloudCover','SurfaceTemperature(C)','HeightSensorBoom(m)','HeightStakes(m)','DepthPressureTransducer(m)','DepthPressureTransducer_Cor(m)','AblationPressureTransducer(mm)','IceTemperature1(C)','IceTemperature2(C)','IceTemperature3(C)','IceTemperature4(C)','IceTemperature5(C)','IceTemperature6(C)','IceTemperature7(C)','IceTemperature8(C)','TiltToEast(d)','TiltToNorth(d)','LatitudeGPS_HDOP<1(ddmm)','LongitudeGPS_HDOP<1(ddmm)','ElevationGPS_HDOP<1(m)','HorDilOfPrecGPS_HDOP<1','LoggerTemperature(C)','FanCurrent(mA)','BatteryVoltage(V)']].tail()
# forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

m.plot(forecast)
plt.show()