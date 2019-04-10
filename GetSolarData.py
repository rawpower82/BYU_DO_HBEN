#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 11:53:27 2019

@author: hunterrawson
"""
import numpy as np
import pandas as pd
import pysolar.solar as sol
from datetime import datetime, timezone, timedelta
import matplotlib.pyplot as plt

# Provo, Utah
latitude = 40.23
longitude = -111.66

data_filename = 'history_export_2019-04-06T18_03_26.csv' # temperature in C, post DST
#'history_export_2019-03-21T21_33_39.csv' # temperature in F, pre and post DST
date_string = data_filename.replace('history_export_','').replace('.csv','')
df = pd.read_csv(data_filename)
years = df['Year'].values
months = df['Month'].values
days = df['Day'].values
hours = df['Hour'].values
Ts = df['Temperature  [2 m above gnd]'].values+273.15 # K #(df['Temperature  [2 m above gnd]'].values-32)*5/9+273.15 # K
Gs = df['Shortwave Radiation  [sfc]'].values # W/m2
n_steps = len(years)

# If DST (March 10 - November 3, 2019), hours=-6; else hours=-7
dst_deltas = np.ones(n_steps)*-6
dst_deltas[np.where(months<3)] = -7
dst_deltas[np.where(months>11)] = -7

Time = np.empty(n_steps)   # hours
Zenith = np.empty(n_steps) # degrees
Azimuth = np.empty(n_steps) # degrees
ClearDirectFlux = np.empty(n_steps) # W/m2

for i in range(n_steps):
    if months[i] == 3:
        if days[i] < 10:
            dst_deltas[i] = -7
    if months[i] == 11:
        if days[i] > 3:
            dst_deltas[i] = -7
    date = datetime(years[i],months[i],days[i],hours[i],0,0,0,timezone(timedelta(hours=dst_deltas[i])))
    if i == 0:
        Time[i] = 0
    else:
        Time[i] = Time[i-1] + hours[1] - hours[0]
    Zenith[i] = 90 - sol.get_altitude(latitude,longitude,date) # degrees; 0 = up; 90 = horizon
    Azimuth[i] = sol.get_azimuth(latitude,longitude,date) # degrees
    ClearDirectFlux[i] = sol.radiation.get_radiation_direct(date, 90-Zenith[i]) # W/m2

headers = ['Time (hr)','Year','Month','Day','Hour','Temperature (K)','Direct Flux (W/m2)',
           'Clear Direct Flux (W/m2)','Zenith (deg from up)','Azimuth (deg from north cw)']
df_save = pd.DataFrame(np.column_stack([Time,years,months,days,hours,Ts,Gs,ClearDirectFlux,
                                        Zenith,Azimuth]),columns=headers)

df_save.to_csv('SolarExport' + date_string + '.csv',index=None)
#%%
plt.figure()
plt.plot(Time,np.ones(n_steps)*90,'k--')
plt.plot(Time,Zenith,label='Zenith')
plt.plot(Time,Azimuth,label='Azimuth')
plt.legend()
plt.xlabel('Time (hours)')
plt.ylabel('Angles (°)')

plt.figure()
plt.plot(Time,ClearDirectFlux,label='Clear Irradiance')
plt.plot(Time,Gs,label='Actual Irradiance')
plt.legend()
plt.xlabel('Time (hours)')
plt.ylabel('Irradiance (W/m$^2$)')
