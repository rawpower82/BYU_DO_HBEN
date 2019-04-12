#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:04:58 2019

@author: hunterrawson
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from gekko import GEKKO
from Solar_Array_Functions import Parameter, SolarPowerMPP, OrientationCorrection, PrintTime
import matplotlib.pyplot as plt
plt.close('all')

# Solar power equation coefficients
aP = 6.5126737778450083E-01
bP = 9.9680804877945284E-01
cP = 1.0391858697506762E+00

# Solar voltage equation coefficients
aV = 1.1834672231963125E+02
bV = 9.9667039815506886E-01
cV = 4.0600537403361850E-02

# Solar module constants
HIT_Single = Parameter(Area_Desired=1.7,Voltage_Desired=70)
A_mod = HIT_Single.SolarPanelArea # (m^2) [area of a single solar module]
V_mod = HIT_Single.V_oc_module # (V) [open circuit voltage of a single solar module]
inverter_eff = 0.9 # [efficiency of DC to AC inverter]

# Roof constants
m2_ft2 = 0.092903  # (m^2/f^2)
RoofDirection = 0 # (degrees) [Roof spine angle clockwise from north - 0 for N-S; 90 for E-W]
RoofPitch = 26.6 # (degrees) [Roof pitch angle]
RoofA_l = 750*m2_ft2 # (m^2) [area of roof - left side]
RoofA_r = 750*m2_ft2 # (m^2) [area of roof - right side]
#Vsol_des = 560 # (V) [target voltage of entire solar array]

# Battery constants
########################################################################
# Charging and Discharging Parameters
dis1=[-1.00e-29, 9.12e-03, 1.593965, 1.0] # Fractional discharging rate
char1=[1.0, 1.0+1e-30, -9.11e-03, -1.593965303] # Fractional charging rate
batt_eff = np.sqrt(0.915) # [one-way efficiency of battery (square to get round trip)]

# Tesla Powerwall Battery
D1=5.0e3   # [W] max continuous discharge rate
C1=15.0e3  # [W] max continuous charge rate
Q1=13.5    # [kWh] storage capacity
# Note that Q1_J is an intermediate

# Tesla ModelS Battery
D2=30.0e3 # [W] max continuous discharge rate
C2=60.0e3 # [W] max continuous charge rate
Q2=87.0   # [kWh] storage capacity  (77)
Q2_J=Q2*3600e3 # (J) [storage capacity]
########################################################################

#%%
# Optimize entire control horizon at once? # Will doing less this prevent solver from knowing when to buy/sell?
# 24 hour timeframe may be sufficient? May be necessary due to large computation demand
# Import data
indices = [192,216]#[0,-1]#[192,216]
df = pd.read_csv('SolarExport2019-04-06T18_03_26.csv')
time = df['Time (hr)'][indices[0]:indices[1]].values-df['Time (hr)'][indices[0]] # (hr)
hours = df['Hour'][indices[0]:indices[1]].values # (hr)
Temperatures = df['Temperature (K)'][indices[0]:indices[1]].values # (K)
DirectFluxes = df['Direct Flux (W/m2)'][indices[0]:indices[1]].values # (W/m2)
ClearDirectFluxes = df['Clear Direct Flux (W/m2)'][indices[0]:indices[1]].values # (W/m2)
Zeniths = df['Zenith (deg from up)'][indices[0]:indices[1]].values # (degrees)
Azimuths =df['Azimuth (deg from north cw)'][indices[0]:indices[1]].values # (degrees)
n = len(time)
p_low = 0.0679 # ($/kWh)
p_high = 0.2278 # ($/kWh)
time24 = np.linspace(0,23,24)
prices24 = np.ones(24)*p_low/3600e3 # ($/J)
prices24[7:11] = p_high/3600e3 # ($/J)
prices24[13:22] = p_high/3600e3 # ($/J)
prices = interp1d(time24,prices24)(hours) # ($/J)
demand24 = np.array([28,26.7,26.3,26.4,26.9,28,29.3,30,30.9,31.8,32.7,31,30.1,29.2,27.6,27.3,27.9,29.1,33.3,35.5,35.2,33,32.3,30.2]) # (kW)
demand24 = demand24*30/sum(demand24)
# These prices were too steep. It should be 30 kWh per day, not 30 kW average
demand = interp1d(time24,demand24)(hours)*1e3 # (W)

# Orientation correction of solar irradiance
LocalFluxL = np.empty(n)
LocalFluxR = np.empty(n)
for i in range(n):
    LocalFluxL[i],LocalFluxR[i],null1,null2 = OrientationCorrection(DirectFluxes[i],Azimuths[i],Zeniths[i],
                                                        RoofDirection,RoofPitch,ViewPlot=False)
#%% Build Model
m = GEKKO()
m.time = time
m.hours = hours
#nm = len(time)*2 # 30 minute intervals
#m.time = np.linspace(time[0],time[-1],nm)
#m.hours = interp1d(time,hours)(m.time)

# Parameters
# T/Gsol will be functions of weather forecast
########################################################################
Tsol_l = m.Param(value=Temperatures) # (K) [solar cell temperature - left side]
Tsol_r = m.Param(value=Temperatures) # (K) [solar cell temperature - right side]
Gsol_l = m.Param(value=LocalFluxL) # (W/m2) [orientation corrected solar irradiation - left side]
Gsol_r = m.Param(value=LocalFluxR) # (W/m2) [orientation corrected solar irradiation - right side]
Pdemand = m.Param(value=demand) # (W) [electricity demand of the house]
# adjust each day with random noise?
price = m.Param(value=prices) # ($/J) [electricity price]
########################################################################
#Tsol_l = m.Param(value=interp1d(time,Temperatures)(m.time)) # (K) [solar cell temperature - left side]
#Tsol_r = m.Param(value=interp1d(time,Temperatures)(m.time)) # (K) [solar cell temperature - right side]
#Gsol_l = m.Param(value=interp1d(time,LocalFluxL)(m.time)) # (W/m2) [orientation corrected solar irradiation - left side]
#Gsol_r = m.Param(value=interp1d(time,LocalFluxR)(m.time)) # (W/m2) [orientation corrected solar irradiation - right side]
#Pdemand = m.Param(value=interp1d(time,demand)(m.time)) # (W) [electricity demand of the house]
## adjust each day with random noise?
#price = m.Param(value=interp1d(time,prices)(m.time)) # ($/J) [electricity price]
########################################################################

# Manipulated Variables
# May need to specify Nm_p so voltage is specified
# Need to place limit on number of panels (within roof area, voltage limits)
# May need to do economic analysis of solar panels separately due to integer solver being slow
# Should we also manipulate the number of Powerwall batteries?
# Need to model use of Tesla during the day (unavailable, SOC drops)
    # Given duty with random variation for Tesla each day? Random deviations from departure and arrival times?
Nm_s_l = m.FV(value=8,lb=1,integer=True) # [number of solar modules in series - left side]
Nm_s_r = m.FV(value=8,lb=1,integer=True) # [number of solar modules in series - right side]
Nm_p_l = m.FV(value=5,lb=1,integer=True) # [number of solar modules in parallel - left side]
Nm_p_r = m.FV(value=5,lb=1,integer=True) # [number of solar modules in parallel - right side]
NN_PW = m.FV(value=1,lb=1,integer=True)  # [number of powerwall batteries]
########################################################################
# This method does not account for battery efficiency, but is much faster
#B_rate1 = m.MV(value=-5e3,lb=-D1*NN_PW,ub=C1*NN_PW) # (W) [rate of charge(+)/discharge(-) of Powerwall batteries]
#B_rate2 = m.MV(value=-25e3,lb=-D2,ub=C2) # (W) [rate of charge(+)/discharge(-) of Tesla ModelS battery]
#B_rate1.STATUS = 1
#B_rate2.STATUS = 1
########################################################################
# This method is quite expensive computationally
B_charge1 = m.MV(value=0,lb=0,ub=C1*NN_PW) # (W) [rate of charge of Powerwall batteries]
B_charge2 = m.MV(value=0,lb=0,ub=C2) # (W) [rate of charge of Tesla ModelS battery]
B_discharge1 = m.MV(value=5e3,lb=0,ub=D1*NN_PW) # (W) [rate of discharge of Powerwall batteries]
B_discharge2 = m.MV(value=25e3,lb=0,ub=D2) # (W) [rate of discharge of Tesla ModelS battery]
B_charge1.STATUS = 1
B_charge2.STATUS = 1
B_discharge1.STATUS = 1
B_discharge2.STATUS = 1
########################################################################

# Intermediates
# Is it possible to reject Psol when Vsol is too low? Maybe make inverter efficiency = f(V)?
# Battery efficiency is not implemented
Psol_l = m.Intermediate(aP*(bP**Tsol_l)*(Gsol_l**cP)*Nm_p_l*Nm_s_l) # [solar array power - left side]
Psol_r = m.Intermediate(aP*(bP**Tsol_r)*(Gsol_r**cP)*Nm_p_r*Nm_s_r) # [solar array power - right side]
Vsol_l = m.Intermediate(aV*(bV**Tsol_l)*(Gsol_l**cV)*Nm_s_l) # (V) [solar array voltage - left side]
Vsol_r = m.Intermediate(aV*(bV**Tsol_r)*(Gsol_r**cV)*Nm_s_r) # (V) [solar array voltage - right side]
Psol_tot = m.Intermediate((Psol_l + Psol_r)*inverter_eff) # (W) [Power from solar array]
Q1_J = m.Intermediate(Q1*NN_PW*3600.0e3) # (J) [storage capacity]
########################################################################
# This method does not account for battery efficiency, but is much faster
#Pbatt = m.Intermediate(-B_rate1-B_rate2) # (W) [Net power flow from(+)/to(-) the batteries]
########################################################################
# This method is quite expensive computationally
Pbatt = m.Intermediate(B_discharge1+B_discharge2-B_charge1-B_charge2) # (W) [Net power flow from(+)/to(-) the batteries]
########################################################################
Pgrid = m.Intermediate(Pdemand-Psol_tot-Pbatt) # (W) [Power bought(+)/sold(-) to the grid]
cost_rate = m.Intermediate(price*Pgrid*3600) # ($/hr) [cost(+)/revenue(-) from buying/selling electricity from/to the grid]

# State variables
SOC1_init = 0.50
SOC2_init = 0.70
SOC1 = m.SV(value=SOC1_init,lb=0,ub=1) # [state of charge of Powerwall batteries]
SOC2 = m.SV(value=SOC2_init,lb=0,ub=1) # [state of charge of Tesla ModelS battery]
cost = m.SV(value=0.00) # ($) [total spent(+)/earned(-) from buying/selling electricity from/to the grid at given time point]

# Equations
########################################################################
# This method does not account for battery efficiency, but is much faster
#m.Equation(SOC1.dt() == B_rate1*3600/Q1_J)#*batt_eff) # multiplying works for charging, but not discharging
#m.Equation(SOC2.dt() == B_rate2*3600/Q2_J)#*batt_eff)
########################################################################
# This method is quite expensive computationally
m.Equation(SOC1.dt() == (B_charge1*batt_eff-B_discharge1*(2-batt_eff))*3600/Q1_J)
m.Equation(SOC2.dt() == (B_charge2*batt_eff-B_discharge2*(2-batt_eff))*3600/Q2_J)
m.Equation(B_charge1*B_discharge1 == 0) # Only one of the two variables can be nonzero
m.Equation(B_charge2*B_discharge2 == 0) # Only one of the two variables can be nonzero
########################################################################
m.Equation(cost.dt() == cost_rate)

# Objectives
p = np.zeros(n)
#p = np.zeros(nm)
p[-1] = 1.0
final = m.Param(value=p)
m.Obj(cost*final)
#m.Obj(cost_rate)
# if multiple objectives are provided, they are summed
q = np.zeros(n)
#q = np.zeros(nm)
q[np.where(m.hours==5)] = 1.0
morning_charge = m.Param(value=q)
#m.Obj(-SOC2*morning_charge*1000000000)
#m.Obj((SOC1-SOC1_init)**2)
#m.Obj((SOC2-SOC2_init)**2)
#m.fix(SOC2,5,1.0) # Charge Tesla by morning
#m.fix(SOC1,23,SOC1_init)
#m.fix(SOC2,23,SOC2_init)
# Max profit / Min bill

# Options
m.options.IMODE = 6
m.options.SOLVER = 3
m.options.NODES = 3
m.options.MV_TYPE = 0

# Solve
m.solver_options = ['max_iter 1250']
m.solve()#disp=False)



#%%
plt.figure()
plt.plot(m.time,np.array(SOC1.value)*100,'-',label='Powerwall')
plt.plot(m.time,np.array(SOC2.value)*100,'--',label='ModelS')
plt.ylabel('SOC (%)')
plt.legend()
plt.grid()

plt.figure()
plt.plot(m.time[:-1],np.array(-Pdemand.value)[:-1]*1e-3,label='Demand')
plt.plot(m.time[:-1],np.array(Psol_tot.value)[:-1]*1e-3,'-.',label='Solar')
plt.plot(m.time[:-1],np.array(Pbatt.value)[:-1]*1e-3,'--',label='Battery')
plt.plot(m.time[:-1],np.array(Pgrid.value)[:-1]*1e-3,':',label='Grid')
plt.ylabel('Power (kW)')
plt.legend()
plt.grid()

plt.figure()
plt.plot(m.time,np.array(cost.value))
plt.ylabel('Cost ($)')
plt.grid()

plt.figure()
plt.plot(m.time[:-1],np.array(B_charge1.value)[:-1]*1e-3,'-',label='Charge 1')
plt.plot(m.time[:-1],np.array(B_charge2.value)[:-1]*1e-3,'--',label='Charge 2')
plt.plot(m.time[:-1],np.array(B_discharge1.value)[:-1]*1e-3,'-',label='Discharge 1')
plt.plot(m.time[:-1],np.array(B_discharge2.value)[:-1]*1e-3,'--',label='Discharge 2')
plt.ylabel('Power (kW)')
plt.legend()
plt.grid()