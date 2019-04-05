#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 15:01:06 2018

@author: hunterrawson
"""
import numpy as np
import pandas as pd
from Solar_Array_Functions import Parameter, SolarPowerMPP

#%% Generate Data

# HIT Module Parameters
HIT_Single = Parameter(Area_Desired=1.7,Voltage_Desired=70)

T = np.linspace(-25,65,10) + 273.15  # (K)
flux = np.arange(0,1200+50,50)      # (W/m^2)
HIT_powers = np.zeros((len(flux),len(T)))
HIT_voltages = np.zeros((len(flux),len(T)))

for i in range(0,len(flux)):
    HIT_Single.G = flux[i]
    for j in range(0,len(T)):
        HIT_Single.T = T[j]
        # return power, current, Vopt, solar_efficiency, solar_section_efficiencies
        HIT_res = SolarPowerMPP([HIT_Single])
        HIT_powers[i,j] = HIT_res[0]   # W
        HIT_voltages[i,j] = HIT_res[2] # V

#%% Save Data
HIT_power_results = np.empty((len(flux)*len(T),3))
HIT_voltage_results = np.empty((len(flux)*len(T),3))
for i in range(0,len(flux)):
    for j in range(0,len(T)):
        HIT_power_results[len(T)*i+j,:] = [flux[i],T[j],HIT_powers[i,j]]
        HIT_voltage_results[len(T)*i+j,:] = [flux[i],T[j],HIT_voltages[i,j]]

HIT_power_DF = pd.DataFrame(HIT_power_results,index=None)
HIT_power_DF.to_csv('SolarData/HIT_Powers_Vs_FluxAndTemp.txt',header=None,index=None,sep=' ')
HIT_voltage_DF = pd.DataFrame(HIT_voltage_results,index=None)
HIT_voltage_DF.to_csv('SolarData/HIT_Voltages_Vs_FluxAndTemp.txt',header=None,index=None,sep=' ')

#%% Results from zunzun.com

## Negative power at 0 W/m2
#def PowerEq(G_in, T_in):
#    # coefficients
#    a0 = -5.7918734663173700E+02
#    a1 = 1.6597789090421043E+03
#    a2 = -1.5955180752254220E-03
#    a3 = -3.9007820282476678E+02
#    a4 = -1.5422796137881490E-02
#    a5 = -1.6222029398078749E-03
#
#    return a0 + (a1 / (1.0 + np.exp(a2 * (G_in + a3 + a4 * T_in + a5 * G_in * T_in))))
def PowerEq(G_in, T_in):
    # coefficients
    a = 6.5126737778450083E-01
    b = 9.9680804877945284E-01
    c = 1.0391858697506762E+00

    return a*(b**T_in)*(G_in**c)

def VoltageEq(G_in, T_in):
    # coefficients
    a = 1.1834672231963125E+02
    b = 9.9667039815506886E-01
    c = 4.0600537403361850E-02

    return a*(b**T_in)*(G_in**c)

powers = np.zeros((len(flux),len(T)))
voltages = np.zeros((len(flux),len(T)))
for i in range(0,len(flux)):
    for j in range(0,len(T)):
        powers[i,j] = PowerEq(flux[i],T[j])
        voltages[i,j] = VoltageEq(flux[i],T[j])

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.close('all')
colors = np.linspace(0,9,10,dtype=int)
while len(colors) < len(flux):
    colors = np.concatenate([colors,colors])

plt.figure()
for i in range(0,len(flux)):
    plt.plot(T-273.15,HIT_powers[i,:],'C' + str(colors[i]) + 'o')
    plt.plot(T-273.15,powers[i,:],'C' + str(colors[i]))
plt.xlabel('Temperature (°C)')
plt.ylabel('Power (W)')

plt.figure()
for i in range(0,len(T)):
    plt.plot(flux,HIT_powers[:,i],'C' + str(colors[i]) + 'o')
    plt.plot(flux,powers[:,i],'C' + str(colors[i]))
plt.xlabel('Solar Irradiance (W/m^2)')
plt.ylabel('Power (W)')

plt.figure()
for i in range(0,len(flux)):
    plt.plot(T-273.15,HIT_voltages[i,:],'C' + str(colors[i]) + 'o')
    plt.plot(T-273.15,voltages[i,:],'C' + str(colors[i]))
plt.xlabel('Temperature (°C)')
plt.ylabel('Voltage (V)')

plt.figure()
for i in range(0,len(T)):
    plt.plot(flux,HIT_voltages[:,i],'C' + str(colors[i]) + 'o')
    plt.plot(flux,voltages[:,i],'C' + str(colors[i]))
plt.xlabel('Solar Irradiance (W/m^2)')
plt.ylabel('Voltage (V)')

n_fine = 1000
flux_fine = np.linspace(flux[0],flux[-1],n_fine)
T_fine = np.linspace(T[0],T[-1],n_fine)
powers_fine = np.zeros((len(flux_fine),len(T_fine)))
voltages_fine = np.zeros((len(flux_fine),len(T_fine)))
for i in range(0,n_fine):
    for j in range(0,n_fine):
        powers_fine[i,j] = PowerEq(flux_fine[i],T_fine[j])
        voltages_fine[i,j] = VoltageEq(flux_fine[i],T_fine[j])

flux_array,T_array = np.meshgrid(flux,T,indexing='ij')
flux_fine_array,T_fine_array = np.meshgrid(flux_fine,T_fine,indexing='ij')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(flux_fine_array,T_fine_array-273.15,powers_fine,cmap='plasma')#,rcount=100,ccount=100)
ax.scatter(flux_array,T_array-273.15,HIT_powers,color='k',marker='o',label='Data')
ax.set_xlabel('Solar Irradiance (W/m^2)')
ax.set_ylabel('Temperature (°C)')
ax.set_zlabel('Power (W)')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(flux_fine_array,T_fine_array-273.15,voltages_fine,cmap='plasma')#,rcount=100,ccount=100)
ax.scatter(flux_array,T_array-273.15,HIT_voltages,color='k',marker='o',label='Data')
ax.set_xlabel('Solar Irradiance (W/m^2)')
ax.set_ylabel('Temperature (°C)')
ax.set_zlabel('Voltage (V)')
