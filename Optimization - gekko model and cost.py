#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 14:02:33 2019

@author: nicoleburchfield
"""


from gekko import GEKKO
import numpy as np
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


#%%

# PRICE DATA
price_data = pd.read_csv('pricing.csv')
time_hr = price_data['time']
hourly_prices = price_data['prices']

#%%

m = GEKKO()

#time array
m.time = np.linspace(0,23,24)   #one day

#--manipulated variables---
xsolar = m.MV(lb = 0, ub = 1)
xbattery = m.MV(lb = 0, ub = 1)
xgrid = m.MV(lb = 0, ub = 1)

#--Control variable--- 
cost = m.CV(value = 0)


#---from data ---
demand = m.Param() #electricy demand data
price = m.Param(value = price_data)

#--intermideiate place holder variable--
used  = m.Param()

#---Equations----
m.Equation(xsolar + xbattery + xgrid == 1.0)
m.Equation(used * (xsolar + xbattery + xgrid) == demand)

#--constraints on solar and battery power used--
m.Equation()    #xsolar*used <= Psolar
m.Equation() #xbattery*used <= Pbattery

#---cost-- 
m.Equation(cost = xgrid*used*price)

'''If xsolar*used < Ps (in the controller) then store the remaining Ps in the 
battery. 


m.Obj(cost)
'''

