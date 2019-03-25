#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 14:31:14 2019

@author: nicoleburchfield
"""

from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import data
#solar radiation data W/m^2 and Temperature Data (deg F)
filename = 'history_export_2019-03-21T21_33_39.csv'
data = pd.read_csv(filename)

#SOURCE: https://www.meteoblue.com/en/weather/archive/export/provo_united-states-of-america_5780026?daterange=2019-03-07+to+2019-03-21&params=&params%5B%5D=11%3B2+m+above+gnd&params%5B%5D=204%3Bsfc&utc_offset=-7&aggregation=hourly&temperatureunit=FAHRENHEIT&windspeedunit=MILE_PER_HOUR

day = data['Day'].values #two weeks of days from march 3 to march 21
hour = data['Hour'].values #from 0 to 23 each day
TempF = data['Temperature  [2 m above gnd]'] #use this as the solar panel temperature
TempC = (TempF - 32) * 5/9
Radiation = data['Shortwave Radiation  [sfc]']


#%%
#Plot

# for day 16 of month 3 (March 16, 2019) (Provo, UT)
plt.subplot(3,1,1)
plt.plot(hour[216:239],TempF[216:239],'r-')
plt.legend(['Temperature (F)'],loc='best')
plt.subplot(3,1,2)
plt.plot(hour[216:239],TempC[216:239],'b-')
plt.legend(['Temperature (C)'],loc='best')
plt.subplot(3,1,3)
plt.plot(hour[216:239],Radiation[216:239],'y--')
plt.xlabel('Time (hours)')
plt.legend(['Solar Irradiation (W/m^2)'],loc='best')
plt.show()

# for the entire 2 weeks (March 7 - 21st, 2019) (Provo, UT)
# NOTE: each day goes from hour 0 to 23 (they all overlap)
plt.subplot(3,1,1)
plt.plot(hour[0:335],TempF[0:335],'r-')
plt.legend(['Temperature (F)'],loc='best')
plt.subplot(3,1,2)
plt.plot(hour[0:335],TempC[0:335],'b-')
plt.legend(['Temperature (C)'],loc='best')
plt.subplot(3,1,3)
plt.plot(hour[0:335],Radiation[0:335],'y--')
plt.xlabel('Time (hours)')
plt.legend(['Radiation (W/m^2)'],loc='best')
plt.show()

#%%
##Nicole: Weather model NOTES
#'''
#Outputs:
#T of solar cells (same as outside)
#Solar radiance flux (normal to the sun rays: Tracking the sun)
#step test in hours. Weather over a day.
#
#need example data based on weather forcast
#'''
#
#def Weather(x):
#    Day,Temp,DewPoint,WindSpeed,SkyCover,Precipitation,Humidity = x
#    SolarIntensity = 1.18*Day + 77.9*Temp + 33.11*DewPoint + 22.8*WindSpeed - 96.9*SkyCover - 49.15*Precipitation - 43.4*Humidity
#    return SolarIntensity
#
##day (0 to 365), Temp in F, windspeed in MPH, everything else in percent (goes from 0 to 1)
#weather_array = 1,1,0.1,5,0.1,0.1,0.1
#f = Weather(weather_array)
#print(f)
#
## for every 100m increase in elevation, the avg temp decreases by 0.7 deg C
##https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=10&ved=2ahUKEwiPnpqcjpThAhWnBjQIHUpnCHwQFjAJegQIBRAC&url=http%3A%2F%2Fwww.ces.fau.edu%2Fnasa%2Fcontent%2Fteacher-materials%2Ftemperature-ppt.pptx&usg=AOvVaw0gEEBUIa_4pCTxDbx-Ajv1
#
## hourly solar radiation data for each day of the year and reference that in the model
##https://www.meteoblue.com/en/weather/archive/export/provo_united-states-of-america_5780026?daterange=2019-03-07+to+2019-03-21&params=&params%5B%5D=204%3Bsfc&utc_offset=-6&aggregation=hourly&temperatureunit=FAHRENHEIT&windspeedunit=MILE_PER_HOUR

