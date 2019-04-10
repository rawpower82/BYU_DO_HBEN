#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:04:58 2019

@author: hunterrawson
"""

import numpy as np
import pandas as pd
from gekko import GEKKO
from Solar_Array_Functions import Parameter, SolarPowerMPP, OrientationCorrection
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

# Tesla Powerwall Battery
D1=5.0e3   # [W] max continuous discharge rate
C1=15.0e3  # [W] max continuous charge rate
Q1=13.5    # [kWh] storage capacity
# Note that Q1_J is an intermediate

# Tesla ModelS Battery
D2=30.0e3 # [W] max continuous discharge rate
C2=60.0e3 # [W] max continuous charge rate
Q2=87.0   # [kWh] storage capacity  (77)
Q2_J=Q2*3600.0e3 # (J) [storage capacity]
########################################################################

#%% Build Model
m = GEKKO()

# Parameters
# T/Gsol will be functions of weather forecast
Tsol_l = m.Param(value=25+273.15) # (K) [solar cell temperature - left side]
Tsol_r = m.Param(value=25+273.15) # (K) [solar cell temperature - right side]
Gsol_l = m.Param(value=800) # (W/m2) [orientation corrected solar irradiation - left side]
Gsol_r = m.Param(value=800) # (W/m2) [orientation corrected solar irradiation - right side]
Pdemand = m.Param(value=30e3) # (W) [electricity demand of the house]
# ($/J) [electricity prices]


# Manipulated Variables
# May need to specify Nm_p so voltage is specified
# Need to place limit on number of panels (within roof area, voltage limits)
# May need to do economic analysis of solar panels separately due to integer solver being slow
# Should we also manipulate the number of Powerwall batteries?
# Need to update bounds on battery rates
# Need to model use of Tesla during the day (unavailable, SOC drops)
    # Given duty with random variation for Tesla each day? Random deviations from departure and arrival times?
Nm_s_l = m.FV(value=8,lb=1,integer=True) # [number of solar modules in series - left side]
Nm_s_r = m.FV(value=8,lb=1,integer=True) # [number of solar modules in series - right side]
Nm_p_l = m.FV(value=5,lb=1,integer=True) # [number of solar modules in parallel - left side]
Nm_p_r = m.FV(value=5,lb=1,integer=True) # [number of solar modules in parallel - right side]
NN_PW = m.FV(value=1,lb=1,integer=True)  # [number of powerwall batteries]
B_rate1 = m.MV(value=-15e3) # (W) [rate of charge(+)/discharge(-) of Powerwall batteries]
B_rate2 = m.MV(value=-15e3) # (W) [rate of charge(+)/discharge(-) of Tesla ModelS battery]
B_rate1.STATUS = 1
B_rate2.STATUS = 1

# State variables
SOC1 = m.SV(value=0.8,lb=0,ub=1) # [state of charge of Powerwall batteries]
SOC2 = m.SV(value=0.8,lb=0,ub=1) # [state of charge of Tesla ModelS battery]

# Controlled Variables
# Do we have any? We aren't really following a set point. Should we have objective functions instead?

# Intermediates
# Is it possible to reject Psol when Vsol is too low? Maybe make inverter efficiency = f(V)?
Psol_l = m.Intermediate(aP*(bP**Tsol_l)*(Gsol_l**cP)*Nm_p_l*Nm_s_l) # [solar array power - left side]
Psol_r = m.Intermediate(aP*(bP**Tsol_r)*(Gsol_r**cP)*Nm_p_r*Nm_s_r) # [solar array power - right side]
Vsol_l = m.Intermediate(aV*(bV**Tsol_l)*(Gsol_l**cV)*Nm_s_l) # (V) [solar array voltage - left side]
Vsol_r = m.Intermediate(aV*(bV**Tsol_r)*(Gsol_r**cV)*Nm_s_r) # (V) [solar array voltage - right side]
Psol_tot = m.Intermediate((Psol_l + Psol_r)*inverter_eff) # (W) [Power from solar array]
Q1_J = m.Intermediate(Q1*NN_PW*3600.0e3) # (J) [storage capacity]
discharge1_max = m.Intermediate(-D1*(dis1[3]/(1+(dis1[1]/(SOC1-dis1[0]))**dis1[2]))) # (W) [maximum discharge rate of Powerwall batteries]
charge1_max = m.Intermediate(C1*(1/(char1[0]+((SOC1-char1[1])/char1[2])**char1[3]))) # (W) [maximum charge rate of Powerwall batteries]
discharge2_max = m.Intermediate(-D2*(dis1[3]/(1+(dis1[1]/(SOC2-dis1[0]))**dis1[2]))) # (W) [maximum discharge rate of Tesla ModelS battery]
charge2_max = m.Intermediate(C2*(1/(char1[0]+((SOC2-char1[1])/char1[2])**char1[3]))) # (W) [maximum charge rate of Tesla ModelS battery]
Pbatt = m.Intermediate(-B_rate1-B_rate2) # (W) [Net power flow from(+)/to(-) the batteries]
Pgrid = m.Intermediate(Pdemand-Psol_tot-Pbatt) # (W) [Power bought(+)/sold(-) to the grid]
#cost = m.Intermediate() # ($) [cost(+)/revenue(-) from buying/selling electricity from/to the grid]

# Have entire control horizon use one bound? # Will this prevent solver from knowing when to buy/sell?
#B_rate1.LOWER = discharge1_max
#B_rate2.LOWER = discharge2_max
#B_rate1.UPPER = charge1_max
#B_rate2.UPPER = charge2_max

# Equations
#m.Equation(discharge1_max <= B_rate1)
#m.Equation(B_rate1 <= charge1_max)
#m.Equation(discharge2_max <= B_rate2)
#m.Equation(B_rate2 <= charge2_max)
m.Equation(SOC1.dt() == B_rate1/Q1_J)
m.Equation(SOC2.dt() == B_rate2/Q2_J)

# Objectives
m.Obj(Pgrid)
# if multiple objectives are provided, they are summed
# m.fix() # Charge Tesla by morning
# Max profit / Min bill

# Options
m.options.IMODE = 6
m.options.SOLVER = 3 # IPOPT
m.options.NODES = 3
m.options.MV_TYPE = 0
#m.options.CV_TYPE = 1 # 1 - ranked objectives; 2 - compromise

#%%
# Import data
indices = [192,216]#[0,-1]
df = pd.read_csv('SolarExport2019-04-06T18_03_26.csv')
time = df['Time (hr)'][indices[0]:indices[1]].values-df['Time (hr)'][indices[0]]
Temperatures = df['Temperature (K)'][indices[0]:indices[1]].values # K
DirectFluxes = df['Direct Flux (W/m2)'][indices[0]:indices[1]].values # W/m2
ClearDirectFluxes = df['Clear Direct Flux (W/m2)'][indices[0]:indices[1]].values # W/m2
Zeniths = df['Zenith (deg from up)'][indices[0]:indices[1]].values # degrees
Azimuths =df['Azimuth (deg from north cw)'][indices[0]:indices[1]].values # degrees
n = len(time)
m.time = time
Tsol_l.value = Temperatures
Tsol_r.value = Temperatures

# Orientation correction of solar irradiance
LocalFluxL = np.empty(n)
LocalFluxR = np.empty(n)
for i in range(n):
    LocalFluxL[i],LocalFluxR[i],null1,null2 = OrientationCorrection(DirectFluxes[i],Azimuths[i],Zeniths[i],
                                                        RoofDirection,RoofPitch,ViewPlot=False)
Gsol_l.value = LocalFluxL
Gsol_r.value = LocalFluxR

# Solve
m.solve()#disp=False)

#%%
plt.figure()
plt.plot(m.time,SOC1.value)
plt.plot(m.time,SOC2.value)

plt.figure()
plt.plot(m.time,np.array(-Pdemand.value)*1e-3,label='Demand')
plt.plot(m.time,np.array(Psol_tot.value)*1e-3,'-.',label='Solar')
plt.plot(m.time,np.array(Pbatt.value)*1e-3,'--',label='Battery')
plt.plot(m.time,np.array(Pgrid.value)*1e-3,':',label='Grid')
plt.ylabel('Power (kW)')
plt.legend()
