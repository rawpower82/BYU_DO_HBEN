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
months = df['Month'][indices[0]:indices[1]].values # (month)
days = df['Day'][indices[0]:indices[1]].values # (day)
hours = df['Hour'][indices[0]:indices[1]].values # (hr)
Temperatures = df['Temperature (K)'][indices[0]:indices[1]].values # (K)
DirectFluxes = df['Direct Flux (W/m2)'][indices[0]:indices[1]].values # (W/m2)
ClearDirectFluxes = df['Clear Direct Flux (W/m2)'][indices[0]:indices[1]].values # (W/m2)
Zeniths = df['Zenith (deg from up)'][indices[0]:indices[1]].values # (degrees)
Azimuths =df['Azimuth (deg from north cw)'][indices[0]:indices[1]].values # (degrees)
n = len(time)
n_days = len(np.unique(days))
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
B2_home_values = np.zeros(n,dtype=int)
B2_home_values[np.where(hours<6)] = 1
B2_home_values[np.where(hours>15)] = 1
#B2_home_values = np.ones(n)
# Orientation correction of solar irradiance
LocalFluxL = np.empty(n)
LocalFluxR = np.empty(n)
for i in range(n):
    LocalFluxL[i],LocalFluxR[i],null1,null2 = OrientationCorrection(DirectFluxes[i],Azimuths[i],Zeniths[i],
                                                        RoofDirection,RoofPitch,ViewPlot=False)

#%% Run solar model for comparison
HIT_l = Parameter(Area_Desired=750*.092903,Voltage_Desired=500,Print=True)
HIT_r = Parameter(Area_Desired=750*.092903,Voltage_Desired=500,Print=True)
Ps = np.empty((n,2)) # (W) [left, right solar power]
Vs = np.empty((n,2)) # (V) [left, right solar voltage]
etas = np.empty((n,2)) # [left, right solar efficiency]
for i in range(n):
    HIT_l.T = Temperatures[i]
    HIT_r.T = Temperatures[i]
    HIT_l.G = LocalFluxL[i]
    HIT_r.G = LocalFluxR[i]
    resultsL = SolarPowerMPP([HIT_l])
    resultsR = SolarPowerMPP([HIT_r])
    Ps[i,0] = resultsL[0]
    Vs[i,0] = resultsL[2]
    etas[i,0] = resultsL[3]
    Ps[i,1] = resultsR[0]
    Vs[i,1] = resultsR[2]
    etas[i,1] = resultsR[3]

#%% Build Model
weights = [5e4] #np.linspace(4,100,196)*1e4
for w in weights:
    print ('Weight:',w)
    try:
        m = GEKKO()
        m.time = time
        m.hours = hours

        # Parameters
        Tsol_l = m.Param(value=Temperatures) # (K) [solar cell temperature - left side]
        Tsol_r = m.Param(value=Temperatures) # (K) [solar cell temperature - right side]
        Gsol_l = m.Param(value=LocalFluxL) # (W/m2) [orientation corrected solar irradiation - left side]
        Gsol_r = m.Param(value=LocalFluxR) # (W/m2) [orientation corrected solar irradiation - right side]
        Pdemand = m.Param(value=demand) # (W) [electricity demand of the house]
        #Pdemand = m.Param(value=demand*(np.random.randn(n)/80+1.0)) # (W) [electricity demand of the house]
        # [random deviation from normal distriubtion with mean 1.0 and std dev 0.0125]
        price = m.Param(value=prices) # ($/J) [electricity price]
        B2_home = m.Param(value=B2_home_values) # (0 or 1) [determines whether Tesla ModelS battery is at home]
        Nm_s_l = m.Param(value=7) # [number of solar modules in series - left side]
        Nm_s_r = m.Param(value=7) # [number of solar modules in series - right side]
        Nm_p_l = m.Param(value=5) # [number of solar modules in parallel - left side]
        Nm_p_r = m.Param(value=5) # [number of solar modules in parallel - right side]
        NN_PW = m.Param(value=1)  # [number of powerwall batteries - not profitable to do more than 1]
        # Need to do economic analysis of solar panels

        # Manipulated Variables
        B_charge1 = m.MV(value=0,lb=0,ub=C1*NN_PW) # (W) [rate of charge of Powerwall batteries]
        B_charge2 = m.MV(value=0,lb=0,ub=C2) # (W) [rate of charge of Tesla ModelS battery]
        B_discharge1 = m.MV(value=0,lb=0,ub=D1*NN_PW) # (W) [rate of discharge of Powerwall batteries]
        B_discharge2 = m.MV(value=0,lb=0,ub=D2) # (W) [rate of discharge of Tesla ModelS battery]
        B_charge1.STATUS = 1
        B_charge2.STATUS = 1
        B_discharge1.STATUS = 1
        B_discharge2.STATUS = 1

        # Intermediates
        # Is it possible to reject Psol when Vsol is too low? Maybe make inverter efficiency = f(V)?
        # Doesn't look necessary (see voltage plot)
        # Battery efficiency is not implemented
        Psol_l = m.Intermediate(aP*(bP**Tsol_l)*(Gsol_l**cP)*Nm_p_l*Nm_s_l) # (W) [solar array power - left side]
        Psol_r = m.Intermediate(aP*(bP**Tsol_r)*(Gsol_r**cP)*Nm_p_r*Nm_s_r) # (W) [solar array power - right side]
        Vsol_l = m.Intermediate(aV*(bV**Tsol_l)*(Gsol_l**cV)*Nm_s_l) # (V) [solar array voltage - left side]
        Vsol_r = m.Intermediate(aV*(bV**Tsol_r)*(Gsol_r**cV)*Nm_s_r) # (V) [solar array voltage - right side]
        Psol_tot = m.Intermediate((Psol_l + Psol_r)*inverter_eff) # (W) [Power from solar array]
        Q1_J = m.Intermediate(Q1*NN_PW*3600.0e3) # (J) [storage capacity]
        Pbatt = m.Intermediate(B_discharge1-B_charge1+(B_discharge2-B_charge2)*B2_home) # (W) [Net power flow from(+)/to(-) the batteries]
        Pgrid = m.Intermediate(Pdemand-Psol_tot-Pbatt) # (W) [Power bought(+)/sold(-) to the grid]
        cost_rate = m.Intermediate(price*Pgrid*3600) # ($/hr) [cost(+)/revenue(-) from buying/selling electricity from/to the grid]

        # State variables
        SOC1_init = 1e-3
        SOC2_init = 1e-3
        SOC1 = m.SV(value=SOC1_init,lb=0,ub=1) # [state of charge of Powerwall batteries]
        SOC2 = m.SV(value=SOC2_init,lb=0,ub=1) # [state of charge of Tesla ModelS battery]
        cost = m.SV(value=0.00) # ($) [total spent(+)/earned(-) from buying/selling electricity from/to the grid at given time point]

        # Equations
        m.Equation(SOC1.dt() == (B_charge1*batt_eff-B_discharge1*(2-batt_eff))*3600/Q1_J)
        m.Equation(SOC2.dt() == (B_charge2*batt_eff*B2_home-B_discharge2*(2-batt_eff))*3600/Q2_J)
        m.Equation(B_charge1*B_discharge1 == 0) # Only one of the two variables can be nonzero
        m.Equation(B_charge2*B_discharge2 == 0) # Only one of the two variables can be nonzero
        m.Equation(cost.dt() == cost_rate)

        # Objectives
        ## multiple objectives are summed
        ## Max profit / Min bill
        p = np.zeros(n)
        p[-1] = 1.0
        final = m.Param(value=p)
        m.Obj(cost*final*100)

        ## Charge Tesla by 6 AM Each morning
        #q = np.zeros(n)
        #q[np.where(m.hours==6)] = 1.0
        #morning_charge = m.Param(value=q)
        #m.Obj(-SOC2*morning_charge*1e4)
        #m.Obj((SOC2-(1.0-1.5e-3))**2*morning_charge*1e12)
        m.fix(SOC2,6,1.0)
        # We could also try SOC2 set points

        ## Tesla returns in afternoon/evening having been used
        r = np.zeros(n)
        r[np.where(m.hours==16)] = 1.0

        #### Confirmation of randn behavior with selecting integers between 13 and 21
        #hrs = np.linspace(12,22,11,dtype=int)
        #rands = np.zeros(11)
        #for i in range(10000):
        #    rand = int(np.random.randn(n_days)[0] + 17.5)
        #    rands[np.where(hrs == rand)] += 1
        #plt.figure()
        #plt.plot(hrs,rands,'o')

        #### Random return hour each day
        #return_hrs = (np.random.randn(n_days) + 17.5).astype(int)
        #return_hrs = return_hrs + 24*np.linspace(0,n_days-1,n_days)
        #for i in range(n_days):
        #    r[np.where(time==return_hrs[i])] = 1.0
        ## [differs each day due to random deviation from normal distriubtion with mean 0.4 and std dev 0.05]

        evening_time = m.Param(value=r)

        ### Random return charge each day
        evening_soc_short = np.ones(n_days)*0.4
        #evening_soc_short = np.random.randn(n_days)/20+0.4
        evening_soc = m.Param(value=np.ones(n)*evening_soc_short[0])
        j = 0
        for i in range(1,n):
            if days[i] != days[i-1]:
                j += 1
        evening_soc.value[i] = evening_soc_short[j]

        #m.Obj((SOC2-0.4)**2*evening_time*6e4)
        m.Obj((SOC2-evening_soc)**2*evening_time*w)#*30e4)

        # We could also try SOC2 set points

        # Options
        m.options.IMODE = 6
        m.options.SOLVER = 3
        m.options.NODES = 4
        m.options.MV_TYPE = 0

        # Solve
        m.solver_options = ['max_iter 1250']
        if len(weights) == 1:
            m.solve()
        else:
            m.solve(disp=False)

        break
    except:
        print ('\tfailed')

#%% Plots
plt.close('all')
if n_days == 1:
    hr_gap = 3
xtix = np.arange(int(m.time[0]),int(m.time[-1]+1),hr_gap)
xtixnames = []
xtixblank = []
for i in range(0,len(xtix)):
    xtixnames.append(PrintTime(xtix[i],minute=False))
    xtixblank.append('')
#plt.xticks(xtix,xtixblank)

# Electricity price/demand plots


plt.figure(figsize=(10,4))
plt.plot(m.time,demand*1e-3,color='darkorange')
plt.ylabel('Electricity Demand (kW)')
plt.grid()
plt.xlim([m.time[0],m.time[-1]])
plt.xticks(xtix,xtixnames)

plt.figure(figsize=(10,4))
plt.plot(m.time,prices*3600e3*100,color='darkorange')
plt.ylabel('Electricity Price ($\cent$/kWh)')
plt.grid()
plt.xlim([m.time[0],m.time[-1]])
plt.xticks(xtix,xtixnames)

# Compare to detailed solar model
plt.figure(figsize=(10,16/3))
plt.subplot(2,1,1)
plt.plot(m.time,np.array(Vsol_l.value),'-',label='Left Twin')
plt.plot(m.time,np.array(Vsol_r.value),'--',label='Right Twin')
plt.plot(m.time,Vs[:,0],'C0o',label='Left')
plt.plot(m.time,Vs[:,1],'C1s',label='Right')
plt.ylabel('Solar Voltage (V)')
plt.legend()
plt.grid()
plt.xlim([m.time[0],m.time[-1]])
plt.xticks(xtix,xtixblank)
plt.subplot(2,1,2)
plt.plot(m.time,np.array(Psol_l.value)/1e3,'-',label='Left Twin')
plt.plot(m.time,np.array(Psol_r.value)/1e3,'--',label='Right Twin')
plt.plot(m.time,Ps[:,0]/1e3,'C0o',label='Left')
plt.plot(m.time,Ps[:,1]/1e3,'C1s',label='Right')
plt.ylabel('Solar Power (kW)')
plt.legend()
plt.grid()
plt.xlim([m.time[0],m.time[-1]])
plt.xticks(xtix,xtixnames)

plt.figure(figsize=(10,8))
plt.subplot(3,1,1)
plt.plot(m.time,np.array(price.value)*3600e3)
plt.ylabel('Price ($/kWh)')
plt.grid()
plt.xlim([m.time[0],m.time[-1]])
plt.xticks(xtix,xtixblank)

B2_home_indices = np.where(B2_home.value>0)[0]
for i in range(1,len(B2_home_indices)):
    if B2_home_indices[i]-B2_home_indices[i-1] > 1:
        B2_home_indices1 = B2_home_indices[:i]
        B2_home_indices1 = np.concatenate([B2_home_indices1,[int(m.time[B2_home_indices[i-1]+1])]])
        B2_home_indices2 = B2_home_indices[i:]
        break

plt.subplot(3,1,2)
plt.plot(m.time[:-1],(np.array(B_charge1.value)[:-1]-np.array(B_discharge1.value)[:-1])*1e-3,'-',label='Powerwall')
plt.plot(m.time[B2_home_indices1],(np.array(B_charge2.value)[B2_home_indices1]*B2_home.value[B2_home_indices1] \
         -np.array(B_discharge2.value)[B2_home_indices1])*1e-3,'C1--',label='ModelS')
plt.plot(m.time[B2_home_indices1[-1]],(np.array(B_charge2.value)[B2_home_indices1[-1]]*B2_home.value[B2_home_indices1[-1]] \
         -np.array(B_discharge2.value)[B2_home_indices1[-1]])*1e-3,'C1o')
plt.plot(m.time[B2_home_indices2[0]],(np.array(B_charge2.value)[B2_home_indices2[0]]*B2_home.value[B2_home_indices2[0]] \
         -np.array(B_discharge2.value)[B2_home_indices2[0]])*1e-3,'C1o')
plt.plot(m.time[B2_home_indices2[:-1]],(np.array(B_charge2.value)[B2_home_indices2[:-1]]*B2_home.value[B2_home_indices2[:-1]] \
         -np.array(B_discharge2.value)[B2_home_indices2[:-1]])*1e-3,'C1--')
plt.ylabel('Battery Rate (kW)')
plt.legend()
plt.grid()
plt.xlim([m.time[0],m.time[-1]])
plt.xticks(xtix,xtixblank)

plt.subplot(3,1,3)
plt.plot(m.time,np.array(SOC1.value)*100,'-',label='Powerwall')
plt.plot(m.time[B2_home_indices1],np.array(SOC2.value)[B2_home_indices1]*100,'--',label='ModelS')
plt.plot(m.time[B2_home_indices1[-1]],np.array(SOC2.value)[B2_home_indices1[-1]]*100,'C1o')
plt.plot(m.time[B2_home_indices2[0]],np.array(SOC2.value)[B2_home_indices2[0]]*100,'C1o')
plt.plot(m.time[B2_home_indices2],np.array(SOC2.value)[B2_home_indices2]*100,'C1--')
plt.ylabel('SOC (%)')
plt.legend()
plt.grid()
plt.xlim([m.time[0],m.time[-1]])
plt.xticks(xtix,xtixnames)

plt.figure(figsize=(10,8))
plt.subplot(3,1,1)
plt.plot(m.time,np.array(price.value)*3600e3)
plt.ylabel('Price ($/kWh)')
plt.grid()
plt.xlim([m.time[0],m.time[-1]])
plt.xticks(xtix,xtixblank)

plt.subplot(3,1,2)
plt.plot(m.time[:-1],np.array(-Pdemand.value)[:-1]*1e-3,label='Demand')
plt.plot(m.time[:-1],np.array(Psol_tot.value)[:-1]*1e-3,'-.',label='Solar')
plt.plot(m.time[:-1],np.array(Pbatt.value)[:-1]*1e-3,'--',label='Battery')
plt.plot(m.time[:-1],np.array(Pgrid.value)[:-1]*1e-3,':',label='Grid')
plt.ylabel('Power (kW)')
plt.legend()
plt.grid()
plt.xlim([m.time[0],m.time[-1]])
plt.xticks(xtix,xtixblank)

plt.subplot(3,1,3)
plt.plot(m.time,np.array(cost.value))
if cost.value[-1] < 0:
    plt.text(0.98,0.5,'Day\'s Revenue: $%.2f' % (-cost.value[-1]),horizontalalignment='right', \
             transform=plt.gca().transAxes,bbox=dict(facecolor='darkorange', alpha=0.8))
plt.ylabel('Cost ($)')
plt.grid()
plt.xlim([m.time[0],m.time[-1]])
plt.xticks(xtix,xtixnames)