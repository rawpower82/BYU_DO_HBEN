# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 12:52:21 2017

@author: hunterrawson
"""

import numpy as np
from scipy.optimize import fsolve,brent
import matplotlib.pyplot as plt
import pandas as pd
from numpy import cos,sin,radians
import time as t
import h5py
#from functools import lru_cache as cache
from matplotlib.ticker import MultipleLocator
#from ApproximateSolarModelParameters import ApproximateModelParameters

import os
file_path = os.getcwd()
os.chdir(file_path)

#print ((0.050*1)*((0.022-0.0171) + 0.0171*1))
#print (1.10*1)

def Series_vs_Parallel_Cells(area,voltage,N_s_guess=91,N_p_guess=770,V_oc_single=1.1,
                             cell_length=0.050,cell_width=0.0171,case_width='none',case_length='none',
                             cell_diagonal='none',cell_type='Alta Devices'):
    '''
    # Calculates number of solar cells to have in series vs. in parallel
    # area - (m^2) [Total area of solar array]
    # voltage - (V) [Desired open circuit voltage of solar array]
    # N_s_guess - [Guess for number of cells in series]
    # N_p_guess - [Guess for number of cells in parallel]
    # V_oc_single - (V/cell) [Open Circuit Voltage of Single Cell]
    # cell_length - (m) [length of a single solar cell]
    # cell_width - (m) [width of a single solar cell]
    # case_width - (m) [width of a single solar cell aperture]
    # case_length - (m) [length of a single solar cell aperture]
    # cell_diagonal - (m) [length of diagonal from one corner of the cell to the other]
    # cell_type - [string containing the name of the solar cell being used, as the caclulation of area changes based on the cell type]
    '''
    if type(case_width) is str:
        case_width = cell_width
    if type(case_length) is str:
        case_length = cell_length

    def area_total(Ns,Np,CellLength,CellWidth,CaseWidth,CaseLength,CellType):
        if 'SunPower' in CellType:
            SolarPanelArea = Ns*Np*CaseWidth*CaseLength # (m^2) (solar array surface area)
        #if CellType == 'Alta Devices':
        else:
            LengthSeries = (CaseLength-CellLength) + CellLength*Ns      # (m) length of solar cells in series
            length_parallel = (CaseWidth-CellWidth) + CellWidth*Np      # (m) [length of solar cells in parallel]
            SolarPanelArea = LengthSeries*length_parallel # (m^2) (solar array surface area)
        return SolarPanelArea

    def functions(X):
        N_s = X[0]
        N_p = X[1]
        F = [0,0]
        F[0] = area - area_total(N_s,N_p,cell_length,cell_width,case_width,case_length,cell_type)
        F[1] = voltage - V_oc_single*N_s
        return F

    NsNp_guess = [N_s_guess,N_p_guess]
    NsNp = fsolve(functions,NsNp_guess)
    N_s = int(NsNp[0]+0.5)  # [Number of solar cells in series] (+0.5 rounds up value)
    N_p = int(NsNp[1]+0.5)  # [Number of strings of solar cells in parallel]

    SolarPanelArea = area_total(N_s,N_p,cell_length,cell_width,case_width,case_length,cell_type) # (m^2) (solar array surface area)
    VoltageOpenCircuit = V_oc_single*N_s          # (V) [open circuit voltage of the array]

    while (SolarPanelArea > area): # loop that ensures Solar Panel area is less than "area" (60 m^2) rather than greater
        N_p -= 1
        SolarPanelArea = area_total(N_s,N_p,cell_length,cell_width,case_width,case_length,cell_type)

    SolarCellArea = cell_length*N_s * cell_width*N_p # (m^2) [solar cell surface area (aperture area not included - used for efficiency calculations)]
    # Area correction for corners of SunPower Cells
    if 'SunPower' in cell_type:
        if type(cell_diagonal) is str:
            print ('SunPower cell diagonal length not given.')
        if cell_length != cell_width:
            print ('Error: SunPower cell length not equal to cell width (shape not square).')
        else:
            diagonal_square = (2*cell_length**2)**(1/2) # m
            diagonal_small_square = diagonal_square - cell_diagonal # m - the diagonal length of a small square (two corners put together)
            length_small_square = (diagonal_small_square**2/2)**(1/2) # m - the side length of a small square (two corners put together)
            area_small_squares = 2*length_small_square**2 # Area of two small squares (4 small triangles) removed from each cell's area
            SolarCellArea -= area_small_squares*N_s*N_p # m^2 - remove area for each cell's corners
    #Ncells = N_s*N_p
    return [N_s,N_p,SolarPanelArea,SolarCellArea,VoltageOpenCircuit]

# 1.4 m chord length
# ~1.626 m air foil length or 1.94
# 22.56 wing length (per wing)
# 1.626/(0.022-0.0171 + 0.0171*95) -> 95 solar cells from front to back?
# 1.94/(0.022-0.0171 + 0.0171*114) -> 114 solar cells from front to back

class Parameter:
    '''
    Definition of a parameter class, which is used to simplify function parameters
    '''
    def __init__(self,V_pv=100,T=-56+273.15,G=1000,TotalMass=425,BattMass=212,PayloadPower=250,SOC_initial=0.25,
                 SOC_Max=1.0,e_densityWhr=350,Area_Desired=60,Voltage_Desired=100,N_s_guess=91,N_p_guess=770,I_pv='none',
                 V_oc_single=1.1,I_sc=0.23,T_coef_V=-0.187,T_coef_I=0.084,a=1.3,I_ph_ref=0.2310,R_p=623,
                 R_s=0.1510,cell_length=0.050,cell_width=0.0171,case_width=0.022,case_length='none',
                 cell_diagonal='none',cell_type='Alta Devices'):
        self.V_pv = V_pv                   # (V) [Voltage panel is operating at - "voltage output of the series coupling" (Vika)]
        self.T = T                         # (K) [temperature of solar panels]
        self.G = G                         # (W/m^2) [solar flux]
        self.TotalMass = TotalMass         # (kg) [total mass of aircraft + payload]
        self.BattMass = BattMass           # (kg) [mass of the battery]
        self.PayloadPower = PayloadPower   # (W) [Power required for payload]
        self.SOC_initial = SOC_initial     # [initial state of charge of the batteries]
        self.SOC_Max = SOC_Max             # [maximum state of charge of the batteries]
        self.e_densityWhr = e_densityWhr   # (W*hr/kg) [energy density of the batteries]
        self.e_densityMJ = e_densityWhr*3600/10**6 # (MJ/kg) [energy density of the batteries]
        [self.N_s,self.N_p,self.SolarPanelArea,self.SolarCellArea,self.SolarVoltageOpenCircuit] = Series_vs_Parallel_Cells(Area_Desired,
                                     Voltage_Desired,N_s_guess,N_p_guess,V_oc_single,cell_length,cell_width,case_width,case_length,cell_diagonal,cell_type)
            # N_s - [Number of solar cells in series]
            # N_p - [Number of strings of solar cells in parallel]
            # SolarPanelArea - (m^2) [solar array surface area]
            # SolarCellArea - (m^2) [solar cell surface area (aperture area not included - used for efficiency calculations)]
            # SolarVoltageOpenCircuit - (V) [open circuit voltage of solar panels in series]
        if type(I_pv) is str:
            self.I_pv = I_ph_ref # (A) [Initial guess for current of cells in series]
        else:
            self.I_pv = I_pv                   # (A) [Initial guess for current of cells in series]
        self.V_oc_single = V_oc_single     # (V/cell) [Open Circuit Voltage of Single Cell]
        self.I_sc = I_sc                   # (A) [Short Circuit Current of Single Cell]
        self.T_coef_V = T_coef_V           # (%/K) [Voltage Temperature Coefficient - percent change per °C from 25°C]
        self.T_coef_I = T_coef_I           # (%/K) [Current Temperature Coefficient - percent change per °C from 25°C]
        self.a = a                         # [Ideality Factor]
        self.I_ph_ref = I_ph_ref           # (A) [Photo Current of a single cell at 25°C and 1000 W/m^2 - Alta Devices]
        self.R_p = R_p                     # (Ohms) {Equivalent parallel resistance of a single cell - Alta Devices]
        self.R_s = R_s                     # (Ohms) [Equivalent series resistance of a single cell - Alta Devices]
#        self.R_p,self.R_s,Io,self.I_ph_ref = ApproximateModelParameters(V_oc_single,Vmpp,Impp,I_sc,1,a,Rs_guess=0.1,Rp_guess=10,Iph_guess='none')
#            # (Ohms) {Equivalent parallel resistance of a single cell - Alta Devices]
#            # (Ohms) [Equivalent series resistance of a single cell - Alta Devices]
#            # (A) [Photo Current of a single cell at 25°C and 1000 W/m^2 - Alta Devices]

def I_Single_String(parameters,UseAltaTempData=False,NREL=False,PrintF=False,ReturnNegative=False):
    '''
    # UseAltaTempData - (bool) [removes temperature dependence from model and uses fit from Alta Devices data instead]
    # NREL - (bool) [uses the NREL-Vika hybrid model]
    '''
    G = parameters.G                          # (W/m^2) [Solar Flux]
    if G <= 0:    # Current will be 0 if flux is negative or 0 - time saver
        I_pv = 0

    else:
        V_pv = parameters.V_pv                    # (V) [Voltage panel is operating at - "voltage output of the series coupling" (Vika)]
        T = parameters.T                          # (K) [Temperature of solar array]
        N_s = parameters.N_s                      # [Number of cells in series]
        I_pv = parameters.I_pv                    # (A) [Initial guess for current of cells in series]
        V_oc = parameters.SolarVoltageOpenCircuit # (V) [Series Open-Circuit Voltage]
        I_sc = parameters.I_sc                    # (A) [Short Circuit Current of Single Cell]
        T_coef_V = parameters.T_coef_V            # (%/K) [Voltage Temperature Coefficient - percent change per °C from 25°C]
        T_coef_I = parameters.T_coef_I            # (%/K) [Current Temperature Coefficient - percent change per °C from 25°C]
        a = parameters.a                          # [Ideality Factor]
        R_p = parameters.R_p                      # (Ohms) {Equivalent parallel resistance of a single cell - Alta Devices]
        R_s = parameters.R_s                      # (Ohms) [Equivalent series resistance of a single cell - Alta Devices]
        I_ph_ref = parameters.I_ph_ref            # (A) [Photo Current of a single cell at 25°C and 1000 W/m^2 - Alta Devices]

        R_p = N_s*R_p # (Ohms) [Equivalent parallel resistance of solar cells in series - Alta Devices]
        R_s = N_s*R_s # (Ohms) [Equivalent series resistance of solar cells in series - Alta Devices]

        q = 1.60217646e-19 # (C) [Charge of an electron]
        k_B = 1.38064852e-23 # (J/K) [Boltzmann constant]
        T_n = 25 + 273.15 # (K)
        Delta_T = T-T_n
        G_n = 1000 # (W/m^2)

        if NREL == True:
            N_p = parameters.N_p  # [Number of strings in parallel]
            R_s = R_s/N_p         # (Ohms) [Equivalent parallel resistance of solar cells in series - Alta Devices]
            #R_p = R_p * G/G_n     # (Ohms) [Equivalent parallel resistance of solar cells in series - Alta Devices]

        V_t = N_s*k_B*T/q # N_s*k_B*T/q (V) [Thermal voltage of solar cells in series (Vika page 22)]

        if UseAltaTempData == True:
            Delta_T = 0
            V_oc = (-0.0015*(T-273.15)+1.1547)*N_s # V (Voltage from data fit given to us by Alta Devices)
            I_sc = (0.0925*(T-273.15)+328.03)/1000 # A (Current from data fit given to us by Alta Devices)
            V_t = N_s*k_B*T_n/q # Changed T to T_n (V) [Thermal voltage of solar cells in series (Vika page 22)]

        K_V = T_coef_V/100*V_oc # (V/K)
        K_I = T_coef_I/100*I_sc # (A/K)
        I_ph = (I_ph_ref + K_I*Delta_T)*(G/G_n) # (A) [Photo Current of solar cells in series - accounts for the difference in temperature and flux from 25°C and 1000 W/m^2]
        I_0 = (I_sc + K_I*Delta_T)/(np.exp((V_oc+K_V*Delta_T)/(a*V_t))-1) # (A)

        def I_Function(X):
            I_pv = X[0]
            F = I_ph - I_0*(np.exp((V_pv+R_s*I_pv)/(V_t*a))-1) - (V_pv + R_s*I_pv)/R_p - I_pv
            #print ('\t',I_ph,-I_0*(np.exp((V_pv+R_s*I_pv)/(V_t*a))-1),-(V_pv + R_s*I_pv)/R_p,-I_pv)
            #print ('\t',R_s*I_pv,V_t*a)
            return F
        Results = fsolve(I_Function,I_pv,full_output=True)
        if Results[2] != 1:
            if I_Function(Results[0]) > 2e-19 and Results[0] >= 0:
                print ('\n\tI_Single_String Failure:',Results[0],1,I_Function(Results[0]))
                print (Results[2:])
        I_pv = float(Results[0])  # (A) [Current from one set of solar cells in series]
        if PrintF == True:
            print (I_Function([I_pv]))
        if ReturnNegative == False:
            if I_pv < 0 or V_pv > V_oc*2: # Blocking diodes prevent negative currents, prevents error with super high currents
                I_pv = 0
    return I_pv
    # Assumptions:
        #  All Identical cells in series
        #  Temperature and irradiance are identical for entire block of series connected cells
            # "In series connected solar cells, the current for the chain is set by the current of the worst performing cell."

def SolarPowerWith_DC_DC_Conversion_Solve(V_pv,high_efficiency,efficiency,parameters_group,UseAltaTempData=False,NREL=False):
    '''
    # Uses a simple DC_DC conversion model. Efficiency is assumed to be constant, though we should definitely check this assumption and it's value
    # V_pv - (V) [Voltage panel is operating at]
    # high_efficiency - [efficiency of the converter when input and output voltages are practically the same]
    # efficiency - [efficiency of the converter in other conditions]
    # parameters_group - [array containing several Parameter objects]
    # UseAltaTempData - (bool) [removes temperature dependence from model and uses fit from Alta Devices data instead]
    # NREL - (bool) [uses the NREL-Vika hybrid model]
    '''
    num_divisions = len(parameters_group)
    voltage_in = V_pv # (V) [voltage entering the converter]
    voltage_out = parameters_group[0].SolarVoltageOpenCircuit # (V) [voltage exiting the converter]
    current_in = 0 # (A) [Current entering the converter from entire solar array]
    for i in range(0,num_divisions):
        parameters_group[i].V_pv = V_pv
        if parameters_group[i].G > 0: # Current will be 0 if flux is negative or 0 - time saver
            current_in += I_Single_String(parameters_group[i],UseAltaTempData,NREL)*parameters_group[i].N_p
    power_in = voltage_in*current_in         # (W) [power entering the converter]
    # If voltages are practically the same, efficiency is very high
    if voltage_in-voltage_out > - 1.0 and voltage_in-voltage_out < 1.0:
        power_out = power_in*high_efficiency  # (W) [power exiting the converter]
    else:
        power_out = power_in*efficiency       # (W) [power exiting the converter]
    return -power_out

def SolarPowerWith_DC_DC_Conversion(V_pv,high_efficiency,efficiency,parameters_group,UseAltaTempData=False,NREL=False):
    '''
    # Uses a simple DC_DC conversion model. Efficiency is assumed to be constant, though we should definitely check this assumption and it's value
    # V_pv - (V) [Voltage panel is operating at]
    # high_efficiency - [efficiency of the converter when input and output voltages are practically the same]
    # efficiency - [efficiency of the converter in other conditions]
    # parameters_group - [array containing several Parameter objects]
    # UseAltaTempData - (bool) [removes temperature dependence from model and uses fit from Alta Devices data instead]
    # NREL - (bool) [uses the NREL-Vika hybrid model]
    '''
    num_divisions = len(parameters_group)
    solar_section_efficiencies = np.zeros(num_divisions)      # (array, unitless) [efficiencies of each solar section]
    voltage_in = V_pv                                         # (V) [voltage entering the converter]
    voltage_out = parameters_group[0].SolarVoltageOpenCircuit # (V) [voltage exiting the converter]
    current_in = 0                                            # (A) [Current entering the converter from entire solar array]
    for i in range(0,num_divisions):
        parameters_group[i].V_pv = V_pv
        if parameters_group[i].G > 0: # Current will be 0 if flux is negative or 0 - time saver
            current_single = I_Single_String(parameters_group[i],UseAltaTempData,NREL)*parameters_group[i].N_p
            current_in += current_single
            solar_section_efficiencies[i] = voltage_in*current_single/(parameters_group[i].G*parameters_group[i].SolarCellArea)
        if parameters_group[0].SolarVoltageOpenCircuit != parameters_group[i].SolarVoltageOpenCircuit:
            print ("Error! Not all strings of solar cells are configured at the same voltage!")
    power_in = voltage_in*current_in         # (W) [power entering the converter]
    # If voltages are practically the same, efficiency is very high
    if voltage_in-voltage_out > - 1.0 and voltage_in-voltage_out < 1.0:
        power_out = power_in*high_efficiency  # (W) [power exiting the converter]
    else:
        power_out = power_in*efficiency       # (W) [power exiting the converter]
    return power_out, power_in, current_in, solar_section_efficiencies

#section1 = Parameter(100,-55+273.15,1000,TotalMass=425,BattMass=212,SOC_initial=0.25,e_densityWhr=350,
#                 Area_Desired=20,Voltage_Desired=100,N_s_guess=91,N_p_guess=770,I_pv=1,V_oc_single=1.1,
#                 I_sc=0.23,T_coef_V=-0.187,T_coef_I=0.084,a=1.3,I_ph_ref=0.2310,R_p=623,R_s=0.1510,
#                 cell_length=0.050,cell_width=0.0171,case_width=0.022)
#start = t.time()
#voltages=np.linspace(0,150,150)
#powers = np.zeros((len(voltages),3))
#for i in range(0,len(voltages)):
#    powers[i,0] = SolarPowerWith_DC_DC_Conversion(voltages[i],1.0,0.85,[section1])[0]
#    powers[i,1] = SolarPowerWith_DC_DC_Conversion(voltages[i],1.0,0.85,[section1])[1]
#end = t.time()
#print(end-start)
#
#plt.figure()
#plt.plot(voltages,powers[:,1],label='Before DC-DC Conversion')
#plt.plot(voltages,powers[:,0],label='After DC-DC Conversion')
#plt.legend()
#plt.xlim([0,150])
#
#voltages = np.linspace(101.8,102,1000)
#powers = np.zeros(len(voltages))
#for i in range(0,len(voltages)):
#    powers[i] = SolarPowerWith_DC_DC_Conversion(voltages[i],1.0,1.0,[section1])[0]
#
#plt.figure()
#plt.plot(voltages,powers)
#plt.xlim([101.8,102])
#plt.xticks([101.8,101.85,101.9,101.95,102.0])
#plt.xlabel("Voltage (V)")
#plt.ylabel("Power (W)")

def SolarPowerMPP(parameters_group,high_efficiency=1.0,efficiency=1.0,UseAltaTempData=False,NREL=False):
    '''
    # parameters_group - [array containing several Parameter objects]
    # high_efficiency - [efficiency of the converter when input and output voltages are practically the same]
    # efficiency - [efficiency of the converter in other conditions]
    # UseAltaTempData - (bool) [removes temperature dependence from model and uses fit from Alta Devices data instead]
    # NREL - (bool) [uses the NREL-Vika hybrid model]
    '''
    num_divisions = len(parameters_group)
    total_flux = 0                                  # (W/m^2) [Flux hitting the entire array]
    total_flux_power = 0                            # (W) [Flux power hitting the entire array]
    for i in range(0,num_divisions):
        total_flux += parameters_group[i].G
        total_flux_power += parameters_group[i].G * parameters_group[i].SolarCellArea

    args = (high_efficiency,efficiency,parameters_group,UseAltaTempData,NREL)
    if total_flux <= 0:
        Vopt = 0
        solar_efficiency = 0
        solar_efficiency_after_DC_DC = 0
        [power_out,power_in,current_in,solar_section_efficiencies] = [0,0,0,np.zeros(num_divisions)]
    else:
        Vopt = brent(SolarPowerWith_DC_DC_Conversion_Solve, args=args, brack=[parameters_group[0].SolarVoltageOpenCircuit*2,1e-12],
                     tol=1.48e-8, full_output=0, maxiter=500) # This uses Brent's Method for optimization
                     # (V) [Optimal operating voltage of the solar array]
        [power_out,power_in,current_in,solar_section_efficiencies] = SolarPowerWith_DC_DC_Conversion(Vopt,high_efficiency,efficiency,parameters_group,UseAltaTempData,NREL)
        if efficiency < 1.0: # Fixes error where the brent function doesn't always catch the high DC_DC conversion efficiency - NOT PERFECT
            [other1,other2,other3,other4] = SolarPowerWith_DC_DC_Conversion(parameters_group[0].SolarVoltageOpenCircuit,high_efficiency,efficiency,parameters_group,UseAltaTempData)
            if other1 > power_out:
                [power_out,power_in,current_in,solar_section_efficiencies] = [other1,other2,other3,other4]
                Vopt = parameters_group[0].SolarVoltageOpenCircuit
        power_out = power_out * 0.983  # (W) [98.3% efficiency - according to peak power trackers from Facebook]
        solar_efficiency = power_in/total_flux_power
        solar_efficiency_after_DC_DC = power_out/total_flux_power
    voltage_out = parameters_group[0].SolarVoltageOpenCircuit # (V) [voltage exiting the converter]
    current_out = power_out/voltage_out       # (A) [current exiting the converter]
    power_lost = power_in-power_out         # (W) [power lost to the converter and MPP tracker]
    return power_out, current_out, voltage_out, solar_efficiency_after_DC_DC, power_in, current_in, Vopt, power_lost, solar_efficiency, solar_section_efficiencies
    '''
    # 0. power_out - (W) [Power exiting the converter]
    # 1. current_out - (A) [Current exiting the converter]
    # 2. voltage_out - (V) [Voltage exiting the converter - should be the system voltage]
    # 3. solar_efficiency_after_DC_DC - (unitless) [Efficiency of solar panels after DC-DC conversion]
    # 4. power_in - (W) [Power entering the converter from the solar array]
    # 5. current_in - (A) [Current entering the converter from entire solar array]
    # 6. Vopt - (V) [Optimal operating voltage of the solar array]
    # 7. power_lost - (W) [Power lost to the converter and MPP tracker]
    # 8. solar_efficiency - (unitless) [Efficiency of solar panels]
    # 9. solar_section_efficiencies - (array, unitless) [efficiencies of each solar section]
    '''

#flux = np.linspace(0,1000,1000)
#current = np.empty(len(flux))
#voltage = np.empty(len(flux))
#solar_array = Parameter(T=-56+273.15)
#for i in range(len(flux)):
#    solar_array.G = flux[i]
#    results = SolarPowerMPP_Multiple([solar_array],2)
#    voltage[i] = results[6]
#    current[i] = results[5]

# Solar Efficiency function
def SolarEfficiency(parameters_group,high_efficiency=1.0,efficiency=1.0,UseAltaTempData=False,NREL=False): # not currently used - calls above function to calculate efficiency
    '''
    # parameters_group - [array containing several Parameter objects]
    # high_efficiency - [efficiency of the converter when input and output voltages are practically the same]
    # efficiency - [efficiency of the converter in other conditions]
    # UseAltaTempData - (bool) [removes temperature dependence from model and uses fit from Alta Devices data instead]
    # NREL - (bool) [uses the NREL-Vika hybrid model]
    '''
#    num_divisions = len(parameters_group)
#
#    total_flux = 0  # (W/m^2) [Flux hitting the entire array]
#    total_flux_power = 0 # (W) [Flux power hitting the entire array]
#    for i in range(0,num_divisions):
#        parameters = parameters_group[i]
#        G = parameters.G                    # (W/m^2) [solar flux at solar array section]
#        area = parameters.SolarCellArea    # (m^2) [area of solar array section]
#        total_flux += G
#        total_flux_power += G*area
#    if total_flux > 0:
    [null0,null1,null2,efficiency_after_DC_DC,null4,null5,null6,null7,efficiency,null9] = SolarPowerMPP(parameters_group,high_efficiency,efficiency,UseAltaTempData,NREL)
    efficiency_after_DC_DC = efficiency_after_DC_DC/0.983
#        efficiency = SolarPowerMPP(parameters_group,high_efficiency,efficiency)[3]/(total_flux_power)
#        efficiency_after_DC_DC = SolarPowerMPP(parameters_group,high_efficiency,efficiency)[0]/(total_flux_power)
#    else:
#        efficiency = 0
#        efficiency_after_DC_DC = 0
    return efficiency, efficiency_after_DC_DC # Efficiency of solar panels, efficiency of solar panels after DC-DC conversion


def SolarPowerMPP_Multiple(parameters_group,N_MPP,high_efficiency=1.0,efficiency=1.0,UseAltaTempData=False,NREL=False):
    '''
    # parameters_group - [array containing several Parameter objects]
    # N_MPP - [even integer specifying the number of MPP trackers on the aircraft]
    # high_efficiency - [efficiency of the converter when input and output voltages are practically the same]
    # efficiency - [efficiency of the converter in other conditions]
    # UseAltaTempData - (bool) [removes temperature dependence from model and uses fit from Alta Devices data instead]
    # NREL - (bool) [uses the NREL-Vika hybrid model]
    '''
    if N_MPP % 2 != 0 and N_MPP != 1:
        print ('Number of MPP Trackers must be even so they can be split between wings!')
        return
    num_divisions = len(parameters_group)
    if N_MPP > num_divisions:
        print ('More MPP Trackers than sections of solar panels!')
        return
    sections_per_wing = int(num_divisions/2)
    sections_per_MPP = int(num_divisions/N_MPP)
    MPP_per_wing = int(N_MPP/2)
    Vopt = np.zeros(num_divisions)
    extra_sections = sections_per_wing-MPP_per_wing*sections_per_MPP # [number of additional sections for the last (back) MPP on each wing]

    power_out = np.zeros(N_MPP)
    power_in = np.zeros(N_MPP)
    current_in = np.zeros(N_MPP)
    solar_section_efficiencies = np.zeros(num_divisions)
    total_flux_power = 0                                # (W) [Flux power hitting the entire array]
    for j in range(0,MPP_per_wing): # Left Wing
        total_flux = 0                                  # (W/m^2) [Flux hitting the entire array]
        for i in range(j*sections_per_MPP,(j+1)*sections_per_MPP):
            #print (j,i)
            total_flux += parameters_group[i].G
            total_flux_power += parameters_group[i].G * parameters_group[i].SolarCellArea
        if j == MPP_per_wing - 1:
            for i in range((j+1)*sections_per_MPP,sections_per_wing):
                #print (j,i)
                total_flux += parameters_group[i].G
                total_flux_power += parameters_group[i].G * parameters_group[i].SolarCellArea
            args = (high_efficiency,efficiency,parameters_group[j*sections_per_MPP:sections_per_wing],UseAltaTempData,NREL)
        else:
            args = (high_efficiency,efficiency,parameters_group[j*sections_per_MPP:(j+1)*sections_per_MPP],UseAltaTempData,NREL)
        if total_flux <= 0:
            Vopt[j*sections_per_MPP:(j+1)*sections_per_MPP] = np.zeros(sections_per_MPP)
            [power_out[j],power_in[j],current_in[j]] = [0,0,0]
            solar_section_efficiencies[j*sections_per_MPP:(j+1)*sections_per_MPP] = np.zeros(sections_per_MPP)
            if j == MPP_per_wing - 1:
                Vopt[(j+1)*sections_per_MPP:sections_per_wing] = np.zeros(extra_sections)
                solar_section_efficiencies[(j+1)*sections_per_MPP:sections_per_wing] = np.zeros(extra_sections)
        else:
            Vopt1 = brent(SolarPowerWith_DC_DC_Conversion_Solve, args=args, brack=[parameters_group[0].SolarVoltageOpenCircuit*2,1e-12],
                     tol=1.48e-8, full_output=0, maxiter=500) # This uses Brent's Method for optimization
                     # (V) [Optimal operating voltage of the solar array sections]
            Vopt[j*sections_per_MPP:(j+1)*sections_per_MPP] = np.ones(sections_per_MPP)*Vopt1
            if j == MPP_per_wing - 1:
                Vopt[(j+1)*sections_per_MPP:sections_per_wing] = np.ones(extra_sections)*Vopt1
                [power_out[j],power_in[j],current_in[j],solar_section_efficiencies[j*sections_per_MPP:sections_per_wing]] = SolarPowerWith_DC_DC_Conversion(Vopt1,
                                                            high_efficiency,efficiency,parameters_group[j*sections_per_MPP:sections_per_wing],UseAltaTempData,NREL)
            else:
                [power_out[j],power_in[j],current_in[j],solar_section_efficiencies[j*sections_per_MPP:(j+1)*sections_per_MPP]] = SolarPowerWith_DC_DC_Conversion(Vopt1,
                                                            high_efficiency,efficiency,parameters_group[j*sections_per_MPP:(j+1)*sections_per_MPP],UseAltaTempData,NREL)
            #if efficiency < 1.0: # Fixes error where the brent function doesn't always catch the high DC_DC conversion efficiency - NOT IMPLEMENTED
            #    [other1,other2,other3,other4] = SolarPowerWith_DC_DC_Conversion(parameters_group[0].SolarVoltageOpenCircuit,high_efficiency,efficiency,parameters_group,UseAltaTempData)
            #    if other1 > power_out:
            #        [power_out,power_in,current_in,solar_section_efficiencies] = [other1,other2,other3,other4]
            #        Vopt = parameters_group[0].SolarVoltageOpenCircuit
            power_out[j] = power_out[j] * 0.983  # (W) [98.3% efficiency - according to peak power trackers from Facebook]
    for j in range(MPP_per_wing,N_MPP): # Right Wing
        total_flux = 0                                  # (W/m^2) [Flux hitting the entire array]
        for i in range(j*sections_per_MPP+extra_sections,(j+1)*sections_per_MPP+extra_sections):
            #print (j,i)
            total_flux += parameters_group[i].G
            total_flux_power += parameters_group[i].G * parameters_group[i].SolarCellArea
        if j == N_MPP - 1:
            for i in range((j+1)*sections_per_MPP+extra_sections,num_divisions):
                #print (j,i)
                total_flux += parameters_group[i].G
                total_flux_power += parameters_group[i].G * parameters_group[i].SolarCellArea
            args = (high_efficiency,efficiency,parameters_group[j*sections_per_MPP+extra_sections:num_divisions],UseAltaTempData,NREL)
        else:
            args = (high_efficiency,efficiency,parameters_group[j*sections_per_MPP+extra_sections:(j+1)*sections_per_MPP+extra_sections],UseAltaTempData,NREL)
        if total_flux <= 0:
            Vopt[j*sections_per_MPP+extra_sections:(j+1)*sections_per_MPP+extra_sections] = np.zeros(sections_per_MPP)
            [power_out[j],power_in[j],current_in[j]] = [0,0,0]
            solar_section_efficiencies[j*sections_per_MPP+extra_sections:(j+1)*sections_per_MPP+extra_sections] = np.zeros(sections_per_MPP)
            if j == N_MPP - 1:
                Vopt[(j+1)*sections_per_MPP+extra_sections:num_divisions] = np.zeros(extra_sections)
                solar_section_efficiencies[(j+1)*sections_per_MPP+extra_sections:num_divisions] = np.zeros(extra_sections)
        else:
            Vopt1 = brent(SolarPowerWith_DC_DC_Conversion_Solve, args=args, brack=[parameters_group[0].SolarVoltageOpenCircuit*2,1e-12],
                     tol=1.48e-8, full_output=0, maxiter=500) # This uses Brent's Method for optimization
                     # (V) [Optimal operating voltage of the solar array sections]
            Vopt[j*sections_per_MPP+extra_sections:(j+1)*sections_per_MPP+extra_sections] = np.ones(sections_per_MPP)*Vopt1
            if j == N_MPP - 1:
                Vopt[(j+1)*sections_per_MPP+extra_sections:num_divisions] = np.ones(extra_sections)*Vopt1
                [power_out[j],power_in[j],current_in[j],solar_section_efficiencies[j*sections_per_MPP+extra_sections:num_divisions]] = SolarPowerWith_DC_DC_Conversion(Vopt1,
                                                            high_efficiency,efficiency,parameters_group[j*sections_per_MPP+extra_sections:num_divisions],UseAltaTempData,NREL)
            else:
                [power_out[j],power_in[j],current_in[j],solar_section_efficiencies[j*sections_per_MPP+extra_sections:(j+1)*sections_per_MPP+extra_sections]] = SolarPowerWith_DC_DC_Conversion(Vopt1,
                                                            high_efficiency,efficiency,parameters_group[j*sections_per_MPP+extra_sections:(j+1)*sections_per_MPP+extra_sections],UseAltaTempData,NREL)
            #if efficiency < 1.0: # Fixes error where the brent function doesn't always catch the high DC_DC conversion efficiency - NOT IMPLEMENTED
            #    [other1,other2,other3,other4] = SolarPowerWith_DC_DC_Conversion(parameters_group[0].SolarVoltageOpenCircuit,high_efficiency,efficiency,parameters_group,UseAltaTempData)
            #    if other1 > power_out:
            #        [power_out,power_in,current_in,solar_section_efficiencies] = [other1,other2,other3,other4]
            #        Vopt = parameters_group[0].SolarVoltageOpenCircuit
            power_out[j] = power_out[j] * 0.983  # (W) [98.3% efficiency - according to peak power trackers from Facebook]

    if total_flux_power <= 0:
        solar_efficiency = 0
        solar_efficiency_after_DC_DC = 0
    else:
        solar_efficiency = sum(power_in)/total_flux_power
        solar_efficiency_after_DC_DC = sum(power_out)/total_flux_power
    voltage_out = parameters_group[0].SolarVoltageOpenCircuit # (V) [voltage exiting the converter]
    current_out = power_out/voltage_out     # (A) [current exiting the converter]
    power_lost = power_in-power_out    # (W) [power lost to the converter and MPP tracker]
    return power_out, current_out, voltage_out, solar_efficiency_after_DC_DC, power_in, current_in, Vopt, power_lost, solar_efficiency, solar_section_efficiencies
    # 0. power_out - (array, W) [Power exiting the converter]
    # 1. current_out - (array, A) [Current exiting the converter]
    # 2. voltage_out - (V) [Voltage exiting the converter - should be the system voltage]
    # 3. solar_efficiency_after_DC_DC - (unitless) [Efficiency of solar panels after DC-DC conversion]
    # 4. power_in - (array, W) [Power entering the converter from the solar array]
    # 5. current_in - (array, A) [Current entering the converter from entire solar array]
    # 6. Vopt - (array, V) [Optimal operating voltage of the solar array]
    # 7. power_lost - (array, W) [Power lost to the converter and MPP tracker]
    # 8. solar_efficiency - (unitless) [Efficiency of solar panels]
    # 9. solar_section_efficiencies - (array, unitless) [efficiencies of each solar section]


############################################ Faster Solar Power Functions ###################################################################################################
def IandV(X,G,N_p,R_p,R_s,V_t,a,I_ph,I_0,MPPeq='simplified analytic'):
    '''
    # X - (A,V,[number of solar sections*2]) [current and voltage guess array]
    # G - (W/m^2,[number of solar sections]) [array of solar fluxes associated with each solar panel section]
    # N_p - (unitless,[number of solar sections]) [Number of strings of solar cells in parallel associated with each solar panel section]
    # R_p - (Ohms) [Equivalent parallel resistance of a single cell - Alta Devices]
    # R_s - (Ohms) [Equivalent series resistance of a single cell - Alta Devices]
    # V_t - (V,[number of solar sections]) [Thermal voltage of solar cells in series (Vika page 22)]
    # a - [Ideality Factor]
    # I_ph - (A,[number of solar sections]) [Photo Current of solar cells in series - accounts for the difference in temperature and flux from 25°C and 1000 W/m^2]
    # I_0 - (A,[number of solar sections])
    # MPPeq - (string) [choose which equation is used to determine the MPP using dP/dV = 0]
    '''
    n = int(len(X)/2)
    I_pv = X[:n]    # (A) [array with all initial guess values for current of cells in series]
    V_pv = X[n:]    # (V) [guess value for optimal voltage the array is operating at - "voltage output of the series coupling" (Vika)]

    #I_pv[I_pv < 0] = 0 # IS THIS CHECK NEEDED? - Blocking diodes prevent negative currents
    zeros = np.zeros(n*2)
    for i in range(0,n):
        if G[i] <= 0:
            if I_pv[i] == 0:
                zeros[i] = 0
                zeros[i+n] = 0
            else:
                zeros[i] = 1000
                zeros[i+n] = 1000
        else:
            zeros[i] = I_ph[i] - I_0[i]*(np.exp((V_pv[i]+R_s*I_pv[i])/(V_t[i]*a))-1) - (V_pv[i] + R_s*I_pv[i])/R_p - I_pv[i]
            if MPPeq == 'analytic':
                zeros[i+n] = I_pv[i]*N_p[i]-N_p[i]*V_pv[i]/(-I_0[i]*R_s/(V_t[i]*a)*np.exp((V_pv[i]+R_s*I_pv[i])/(V_t[i]*a))-R_s/R_p-1)*(-I_0[i]/(V_t[i]*a)*np.exp((V_pv[i]+R_s*I_pv[i])/(V_t[i]*a))-1/R_p) # dP/dV by analytic sensitivity equations
            elif MPPeq == 'simplified analytic':
                zeros[i+n] = 1 + (R_s - N_p[i]*V_pv[i]/(I_pv[i]*N_p[i])) * (I_0[i]/(V_t[i]*a) * np.exp((V_pv[i] + R_s*I_pv[i])/(V_t[i]*a)) + 1/R_p) # dP/dV (simplified) by analytic sensitivity equations
    return zeros

def I_pv_func(G):
    # coefficients
    b = 2.1060014317386282E-04
    c = 3.3198514015588199E+00
    d = 8.0551718437898307E+02
    Offset = -1.6869254344043364E-03
    if G == 0:
        G = 1e-20
    return np.log(c + np.exp(b * d * G)) / d + Offset

def Vopt_func(G):
    # coefficients
    a = 1.0440115912681620E+01
    b = 1.1305539991358546E+02
    c = 9.0087505478550298E+00
    d = 4.3100546185107902E-01
    if G == 0:
        G = 1e-20
    return (a - b) / (1.0 + np.exp((G - c) / (d * G))) + b

def SolarPowerMPP_Direct(parameters_group,num_divisions,T,G,SolarPanelAreas,N_s,N_p,R_p,R_s,V_oc,I_sc,T_coef_V,T_coef_I,a,I_ph_ref,q,k_B,T_n,G_n,
                         K_V,K_I,I_pv_guess,Voltage_guess,high_efficiency=1.0,efficiency=1.0,UseAltaTempData=False,MPPeq='simplified analytic',UseGuessValueFunctions=False):
    '''
    # parameters_group - [array containing several Parameter objects]
    # num_divisions - [number of solar sections]
    # T - (K,[number of solar sections]) [array of solar panel temperatures associated with each solar panel section]
    # G - (W/m^2,[number of solar sections]) [array of solar fluxes associated with each solar panel section]
    # high_efficiency - [efficiency of the converter when input and output voltages are practically the same]
    # SolarPanelAreas - (m^2,[number of solar sections]) [array of the areas of each solar panel section]
    # N_s - (unitless) [Number of cells in series - This is not an array because the number of cells in series must be the same for all strings if the open circuit voltage is the same for all strings]
    # N_p - (unitless,[number of solar sections]) [Number of strings of solar cells in parallel associated with each solar panel section]
    # R_p - (Ohms) [Equivalent parallel resistance of a single cell - Alta Devices]
    # R_s - (Ohms) [Equivalent series resistance of a single cell - Alta Devices]
    # V_oc - (V) [Series Open-Circuit Voltage]
    # I_sc - (A) [Short Circuit Current of Single Cell]
    # T_coef_V - (%/K) [Voltage Temperature Coefficient - percent change per °C from 25°C]
    # T_coef_I - (%/K) [Current Temperature Coefficient - percent change per °C from 25°C]
    # a - [Ideality Factor]
    # I_ph_ref - (A) [Photo Current of a single cell at 25°C and 1000 W/m^2 - Alta Devices]
    # q - (C) [Charge of an electron]
    # k_B - (J/K) [Boltzmann constant]
    # T_n - (K)
    # G_n - (W/m^2)
    # K_V - (V/K)
    # K_I - (A/K)
    # I_pv_guess - (A) [guess value for the current of a single string of solar cells in series]
    # Voltage_guess - (V) [guess value for the optimal operating voltage- "voltage output of the series coupling" (Vika)]
    # efficiency - [efficiency of the converter in other conditions]
    # UseAltaTempData - (bool) [removes temperature dependence from model and uses fit from Alta Devices data instead]
    # MPPeq - (string) [choose which equation is used to determine the MPP using dP/dV = 0]
    # UseGuessValueFunctions - (bool) [used for toggling the use of functions fit from data for finding guess values]
    '''
    total_flux = sum(G)                               # (W/m^2) [Flux hitting the entire array]
    voltage_out = parameters_group[0].SolarVoltageOpenCircuit # (V) [voltage exiting the converter]

    if total_flux <= 0:
        Vopt = 0
        solar_efficiency = 0
        solar_efficiency_after_DC_DC = 0
        [power_out,power_in,currents_single,solar_section_efficiencies] = [0,0,np.zeros(num_divisions),np.zeros(num_divisions)]
    else:
        Delta_T = T-T_n
        V_t = N_s*k_B*T/q # (V,[number of solar sections]) [Thermal voltage of solar cells in series (Vika page 22)]
        if UseAltaTempData == True:
            Delta_T = 0
            V_oc = (-0.0015*(T-273.15)+1.1547)*N_s # V (Voltage from data fit given to us by Alta Devices)
            I_sc = (0.0925*(T-273.15)+328.03)/1000 # A (Current from data fit given to us by Alta Devices)
            V_t = N_s*k_B*T_n/q # Changed T to T_n (V) [Thermal voltage of solar cells in series (Vika page 22)]
            K_V = T_coef_V/100*V_oc # (V/K)
            K_I = T_coef_I/100*I_sc # (A/K)
        I_ph = (I_ph_ref + K_I*Delta_T)*(G/G_n) # (A,[number of solar sections]) [Photo Current of solar cells in series - accounts for the difference in temperature and flux from 25°C and 1000 W/m^2]
        I_0 = (I_sc + K_I*Delta_T)/(np.exp((V_oc+K_V*Delta_T)/(a*V_t))-1) # (A,[number of solar sections])

        total_flux_power = sum(G * SolarPanelAreas)   # (W) [Flux power hitting the entire array]

        arg = (G,N_p,R_p,R_s,V_t,a,I_ph,I_0,MPPeq)
        guesses = np.zeros(num_divisions*2)
        if T[0] == -56+273.15 and T[0] == T[-1] and UseGuessValueFunctions == True:
            for i in range(num_divisions):
                guesses[i] = I_pv_func(G[i])                                   # (A) [array with all initial guess values for current of cells in series]
                guesses[i+num_divisions] = Vopt_func(G[i])                     # (V) [guess value for optimal voltage each section is operating at - "voltage output of the series coupling" (Vika)]
        else:
            if UseGuessValueFunctions == True:
                print ('\tZunzun fit not used!')
            guesses[:num_divisions] = np.ones(num_divisions)*I_pv_guess        # (A) [array with all initial guess values for current of cells in series]
            guesses[num_divisions:] = np.ones(num_divisions)*Voltage_guess     # (V) [guess value for optimal voltage each section is operating at - "voltage output of the series coupling" (Vika)]
        for i in range(0,num_divisions):
            if G[i] <= 0:
                guesses[i] = 0
                guesses[i+num_divisions] = 0
        Results_w_message = fsolve(IandV,guesses,args=arg,full_output=True)
        Results = Results_w_message[0]
        if Results_w_message[2] != 1:
            print (Results_w_message[2:])

        Vopt = Results[num_divisions:] # (V,[number of solar sections]) [Optimal operating voltages of the solar array sections]
        #voltage_in = Vopt                                        # (V) [voltage entering the converter]
        solar_section_efficiencies = np.zeros(num_divisions)      # (array, unitless) [efficiencies of each solar section]
        currents_single = Results[:num_divisions] * N_p                      # (A,[number of solar sections]) [currents from each solar section]
        ###################currents_single[currents_single < 0] = 0 # ONLY COMMENTED OUT FOR TROUBLESHOOTING
        # current_in = sum(currents_single)    # (A) [Current entering the converter from entire solar array] - Not valid when each section is at a different voltage
        for i in range(0,num_divisions):
            if G[i] > 0:
                solar_section_efficiencies[i] = Vopt[i]*currents_single[i]/(G[i]*SolarPanelAreas[i])

        power_in = sum(Vopt*currents_single)                # (W) [power entering the converter]
        power_out = power_in * 0.983  # (W) [98.3% efficiency - according to peak power trackers from Facebook]
        # =============================================================================
        # DC-DC conversion efficiency not used for now
        # =============================================================================
        ## If voltages are practically the same, efficiency is very high
        #if Vopt-voltage_out > - 1.0 and Vopt-voltage_out < 1.0:
        #    power_out = power_in*high_efficiency  # (W) [power exiting the converter]
        #else:
        #    power_out = power_in*efficiency       # (W) [power exiting the converter]
        #
        ## This won't work - need to update for new functions
        #if efficiency < 1.0: # Fixes error where the brent function doesn't always catch the high DC_DC conversion efficiency - NOT PERFECT
        #    [other1,other2,other3,other4] = SolarPowerWith_DC_DC_Conversion(voltage_out,high_efficiency,efficiency,parameters_group,UseAltaTempData)
        #    if other1 > power_out:
        #        [power_out,power_in,current_in,solar_section_efficiencies] = [other1,other2,other3,other4]
        #        Vopt = voltage_out

        solar_efficiency = power_in/total_flux_power
        solar_efficiency_after_DC_DC = power_out/total_flux_power

    current_out = power_out/voltage_out      # (A) [current exiting the converter]
    power_lost = power_in-power_out         # (W) [power lost to the converter and MPP tracker]
    return power_out, current_out, voltage_out, solar_efficiency_after_DC_DC, power_in, currents_single, Vopt, power_lost, solar_efficiency, solar_section_efficiencies
    '''
    # power_out - (W) [Power exiting the converter]
    # current_out - (A) [Current exiting the converter]
    # voltage_out - (V) [Voltage exiting the converter - should be the system voltage]
    # solar_efficiency_after_DC_DC - (unitless) [Efficiency of solar panels after DC-DC conversion]
    # power_in - (W) [Power entering the converter from the solar array]
    # currents_single - (A,[number of solar sections]) [Current entering the converter from a single section]
    # Vopt - (V,[number of solar sections]) [Optimal operating voltage of the solar array]
    # power_lost - (W) [Power lost to the converter and MPP tracker]
    # solar_efficiency - (unitless) [Efficiency of solar panels]
    # solar_section_efficiencies - (unitless,[number of solar sections]) [efficiencies of each solar section]
    '''


def Simulation_Direct(time,temperatures,fluxes,parameters_group,I_pv_guess=0.05,Voltage_guess=101,high_efficiency=1.0,efficiency=1.0,UseAltaTempData=False,MPPeq='simplified analytic',Print=False,UseGuessValueFunctions=False):
    '''
    # time - (hours) [linear array of times]
    # temperatures - (K,[length of time array, number of solar sections]) [array of solar panel temperatures associated with the time array and each solar panel section]
    # fluxes - (W/m^2,[length of time array, number of solar sections]) [array of solar fluxes associated with the time array and each solar panel section]
    # parameters_group - [array containing several Parameter objects]
    # I_pv_guess - (A) [guess value for the current of a single string of solar cells]
    # Voltage_guess - (V) [guess value for the optimal operating voltage- "voltage output of the series coupling" (Vika)]
    # high_efficiency - [efficiency of the converter when input and output voltages are practically the same]
    # efficiency - [efficiency of the converter in other conditions]
    # UseAltaTempData - (bool) [removes temperature dependence from model and uses fit from Alta Devices data instead]
    # MPPeq - (string) [choose which equation is used to determine the MPP using dP/dV = 0]
    # Print - (bool) [used for toggling whether periodic updates on simulation time are printed]
    # UseGuessValueFunctions - (bool) [used for toggling the use of functions fit from data for finding guess values]
    '''
    num_divisions = len(parameters_group)   # number of solar sections
    n = len(time)
    # Time Independent Arrays
    SolarPanelAreas = np.empty(num_divisions) # (m^2) [Solar panel section areas]
    N_p = np.empty(num_divisions)             # [Number of strings in parallel]
    for i in range(0,num_divisions):
        SolarPanelAreas[i] = parameters_group[i].SolarCellArea
        N_p[i] = parameters_group[i].N_p

    # Constants (All Independent of Time)
    N_s = parameters_group[0].N_s                      # [Number of cells in series - This is not an array because the number of cells in series must be the same for all strings if the open circuit voltage is the same for all strings]
    R_p = parameters_group[0].R_p                      # (Ohms) [Equivalent parallel resistance of a single cell - Alta Devices]
    R_s = parameters_group[0].R_s                      # (Ohms) [Equivalent series resistance of a single cell - Alta Devices]
    V_oc = parameters_group[0].SolarVoltageOpenCircuit # (V) [Series Open-Circuit Voltage]
    I_sc = parameters_group[0].I_sc                    # (A) [Short Circuit Current of Single Cell]
    T_coef_V = parameters_group[0].T_coef_V            # (%/K) [Voltage Temperature Coefficient - percent change per °C from 25°C]
    T_coef_I = parameters_group[0].T_coef_I            # (%/K) [Current Temperature Coefficient - percent change per °C from 25°C]
    a = parameters_group[0].a                          # [Ideality Factor]
    I_ph_ref = parameters_group[0].I_ph_ref            # (A) [Photo Current of a single cell at 25°C and 1000 W/m^2 - Alta Devices]
    q = 1.60217646e-19   # (C) [Charge of an electron]
    k_B = 1.38064852e-23 # (J/K) [Boltzmann constant]
    T_n = 25 + 273.15    # (K)
    G_n = 1000           # (W/m^2)

    R_p = N_s*R_p # (Ohms) [Equivalent parallel resistance of solar cells in series - Alta Devices]
    R_s = N_s*R_s # (Ohms) [Equivalent series resistance of solar cells in series - Alta Devices]
    K_V = T_coef_V/100*V_oc # (V/K)
    K_I = T_coef_I/100*I_sc # (A/K)

    # Initialize Arrays
    power_out = np.empty(n)
    current_out = np.empty(n)
    voltage_out = np.empty(n)
    solar_efficiency_after_DC_DC = np.empty(n)
    power_in = np.empty(n)
    currents_single = np.empty((n,num_divisions))
    Vopt = np.empty((n,num_divisions))
    power_lost = np.empty(n)
    solar_efficiency = np.empty(n)
    solar_section_efficiencies = np.empty((n,num_divisions))

    midt = t.time()
    for i in range(0,len(time)):
        if Print == True:
            if i % 10 == 0:
                print ('\titeration: %3i (%.4f sec)' % (i,t.time()-midt))
            midt = t.time()
            if i == int(n/4):
                print ("\t****** 25 %  Complete ******")
            if i == int(n/2):
                print ("\t****** 50 %  Complete ******")
            if i == int(3*n/4):
                print ("\t****** 75 %  Complete ******")
        #for j in range(0,num_divisions): # These can be commented out when not troubleshooting
        #    parameters_group[j].T = temperatures[i,j] # These can be commented out when not troubleshooting
        #    parameters_group[j].G = fluxes[i,j] # These can be commented out when not troubleshooting
        [power_out[i], current_out[i], voltage_out[i], solar_efficiency_after_DC_DC[i],
         power_in[i], currents_single[i,:], Vopt[i,:], power_lost[i], solar_efficiency[i],
         solar_section_efficiencies[i,:]] = SolarPowerMPP_Direct(parameters_group,num_divisions,temperatures[i,:],fluxes[i,:],SolarPanelAreas,N_s,N_p,R_p,R_s,V_oc,I_sc,
                                         T_coef_V,T_coef_I,a,I_ph_ref,q,k_B,T_n,G_n,K_V,K_I,I_pv_guess,Voltage_guess,high_efficiency,efficiency,UseAltaTempData,MPPeq,UseGuessValueFunctions)
    return power_out, current_out, voltage_out, solar_efficiency_after_DC_DC, power_in, currents_single, Vopt, power_lost, solar_efficiency, solar_section_efficiencies
    '''
    # power_out - (W) [Power exiting the converter]
    # current_out - (A) [Current exiting the converter]
    # voltage_out - (V) [Voltage exiting the converter - should be the system voltage]
    # solar_efficiency_after_DC_DC - (unitless) [Efficiency of solar panels after DC-DC conversion]
    # power_in - (W) [Power entering the converter from the solar array]
    # currents_single - (A,[number of solar sections]) [Current entering the converter from a single section]
    # Vopt - (V,[length of time array, number of solar sections]) [Optimal operating voltage of the solar array]
    # power_lost - (W) [Power lost to the converter and MPP tracker]
    # solar_efficiency - (unitless) [Efficiency of solar panels]
    # solar_section_efficiencies - (unitless,[length of time array, number of solar sections]) [efficiencies of each solar section]
    '''

#%%
################################ Analyze different solar panel parameters ################################

"""
2016 Alta Devices
    V_oc_single=1.09,I_sc=0.246,T_coef_V=-0.187,T_coef_I=0.084,a=1.3,I_ph_ref=0.2520,R_p=706,R_s=0.1510,cell_length=0.050?,cell_width=0.0170?,case_width=0.022?

2017 Alta Devices
    V_oc_single=1.1,I_sc=0.23,T_coef_V=-0.187,T_coef_I=0.084,a=1.3,I_ph_ref=0.2310,R_p=623,R_s=0.1510,cell_length=0.050,cell_width=0.0171,case_width=0.022
    Rp = 435.680945841
    Rs = 0.122561471234
    Iph = 0.230064701335

SunPower Cells
    V_oc_single=0.684,I_sc=6.26,T_coef_V=-0.27,T_coef_I=0.05,a=1.3,I_ph_ref=6.35,R_p=11,R_s=0.1510,cell_length=0.125,cell_width=0.125,case_width=0.125
    (Averaged from sources below)
    https://www.renogy.com/template/files/Specifications/SPR-OS-Flex-100.pdf
        	Dimensions: 1165 x 569 x 20 mm (outer sections not panels?)
        	Cell type: 32 Prime monocrystalline SunPower IBC cells
        	T_coef_V: -0.0589 V/°C
        	T_coef_I: 0.0026 A/°C
        	T_coef_P: -0.35 %/°C
        	Max system voltage of 45 V
         Ncells = 32
         RP = 11
        	Rs = 1.15
        	110:
        		Voc = 21.7 # V
        		Isc = 6.3 # A
        		Iph = 6.4 # A
        	100:
        		Voc = 21 # V
        		Isc = 6.2 # A
        		Iph = 6.3 # A

    http://www.sun-life.com.ua/doc/sunpower%20C60.pdf
        Ncells = 1
        Dimensions: 125 mm x 125 mm
        T_coef_V: -0.0018 V/°C
        T_coef_P: -0.32 %/°C
        C60-G:
        		Voc = 0.682 # V
        		Isc = 6.24 # A
        		RP = 8.42031866754
        		Rs = 0.00188561815287
        		Iph = 6.24139736484
        C60-H:
        		Voc = 0.684 # V
        		Isc = 6.26 # A
        		RP = 12.2865939783
        		Rs = 0.00170362278545
        		Iph = 6.26086799309
        C60-I:
        		Voc = 0.686 # V
        		Isc = 6.27 # A
        		RP = 20.9283073857
        		Rs = 0.00133322456377
        		Iph = 6.27039942638
        C60-J:
        		Voc = 0.687 # V
        		Isc = 6.28 # A
        		RP = 112.376736161
        		Rs = 0.00135952869975
        		Iph = 6.28007597516

    https://www.solbian.eu/img/cms/PDF/Datasheet%20SP.pdf
        Isc = 6.0 # A
        T_coef_V = -0.27 %/°C
        	T_coef_I = 0.05 %/°C
        	T_coef_P = -0.38 %/°C
        	Operating temp: -40°C to 85 °C
        	Rp = 11
        	Rs = 1.150
        	SP 144
        		Voc = 30.0 # V
        		Iph = 6.0010
              Ncells = 44
        	SP 130
        		Voc = 27.3 # V
        		Iph = 6.1000
              Ncells = 40
        	SP 118 L (or Q)
        		Voc = 24.5 # V
        		Iph = 6.1000
              Ncells = 36
        	SP 104
        		Voc = 21.8 # V
        		Iph = 6.1000
              Ncells = 32
        	SP 78
        		Voc = 16.4 # V
        		Iph = 6.1000
              Ncells = 24
        	SP 52 L
        		Voc = 10.9 # V
        		Iph = 6.1000
              Ncells = 16
    """
#if __name__ == "__main__":
#    # 2017 Alta Devices - Old parameters
#    I_ph_ref_AltaOld=0.2310
#    R_p_AltaOld=623
#    R_s_AltaOld=0.1510
#
#    # 2017 Alta Devices - New parameters
#    V_oc_single_Alta=1.1
#    I_sc_Alta=0.23
#    T_coef_V_Alta=-0.187
#    T_coef_I_Alta=0.084
#    I_ph_ref_Alta=0.230064701335
#    R_p_Alta=435.680945841
#    R_s_Alta=0.122561471234
#    cell_length_Alta=0.050
#    cell_width_Alta=0.0171
#    case_width_Alta=0.022
#
#    # SunPower Cells - G
#    V_oc_single_SPG=0.682
#    I_sc_SPG=6.24
#    T_coef_V_SPG=-0.26
#    T_coef_I_SPG=0.05
#    I_ph_ref_SPG=6.24139736484
#    R_p_SPG=8.42031866754
#    R_s_SPG=0.00188561815287
#    cell_length_SPG=0.125
#    case_length_SPG=cell_length_SPG + 0.002
#    cell_width_SPG=0.125
#    case_width_SPG=cell_width_SPG + 0.002
#    cell_diagonal_SP=0.160
#
#    # SunPower Cells - H
#    V_oc_single_SPH=0.684
#    I_sc_SPH=6.26
#    T_coef_V_SPH=-0.26
#    T_coef_I_SPH=0.05
#    I_ph_ref_SPH=6.26086799309
#    R_p_SPH=12.2865939783
#    R_s_SPH=0.00170362278545
#    cell_length_SPH=0.125
#    case_length_SPH=cell_length_SPH + 0.002
#    cell_width_SPH=0.125
#    case_width_SPH=cell_width_SPH + 0.002
#
#    # SunPower Cells - I
#    V_oc_single_SPI=0.686
#    I_sc_SPI=6.27
#    T_coef_V_SPI=-0.26
#    T_coef_I_SPI=0.05
#    I_ph_ref_SPI=6.27039942638
#    R_p_SPI=20.9283073857
#    R_s_SPI=0.00133322456377
#    cell_length_SPI=0.125
#    case_length_SPI=cell_length_SPI + 0.002
#    cell_width_SPI=0.125
#    case_width_SPI=cell_width_SPI + 0.002
#
#    # SunPower Cells - J
#    V_oc_single_SPJ=0.687
#    I_sc_SPJ=6.28
#    T_coef_V_SPJ=-0.26
#    T_coef_I_SPJ=0.05
#    I_ph_ref_SPJ=6.28007597516
#    R_p_SPJ=112.376736161
#    R_s_SPJ=0.00135952869975
#    cell_length_SPJ=0.125
#    case_length_SPJ=cell_length_SPJ + 0.002
#    cell_width_SPJ=0.125
#    case_width_SPJ=cell_width_SPJ + 0.002
#
#    # 2017 Alta Devices - NREL-Vika Hybrid parameters
#    a_Alta_hyb = 1.30028729241
#    I_ph_ref_Alta_hyb=0.23006540709
#    R_p_Alta_hyb=436.642922263
#    R_s_Alta_hyb=0.124171926282
#
#    # SunPower Cells - H - NREL-Vika Hybrid parameters
#    a_SPH_hyb = 1.2936774812
#    I_ph_ref_SPH_hyb=6.26095039124
#    R_p_SPH_hyb=11.9400116102
#    R_s_SPH_hyb=0.00181272324012
#
#    SingleCellAltaOld = Parameter(Area_Desired=0.0012,Voltage_Desired=1.1,N_s_guess=1,N_p_guess=1,
#                 V_oc_single=V_oc_single_Alta,I_sc=I_sc_Alta,T_coef_V=T_coef_V_Alta,T_coef_I=T_coef_I_Alta,
#                 I_ph_ref=I_ph_ref_AltaOld,R_p=R_p_AltaOld,R_s=R_s_AltaOld,cell_length=cell_length_Alta,cell_width=cell_width_Alta,case_width=case_width_Alta)
#    SingleCellAlta = Parameter(Area_Desired=0.0012,Voltage_Desired=1.1,N_s_guess=1,N_p_guess=1,
#                 V_oc_single=V_oc_single_Alta,I_sc=I_sc_Alta,T_coef_V=T_coef_V_Alta,T_coef_I=T_coef_I_Alta,
#                 I_ph_ref=I_ph_ref_Alta,R_p=R_p_Alta,R_s=R_s_Alta,cell_length=cell_length_Alta,cell_width=cell_width_Alta,case_width=case_width_Alta)
#    SingleCellSunPowerG = Parameter(Area_Desired=0.0173,Voltage_Desired=0.69,N_s_guess=1,N_p_guess=1,#I_pv=1,
#                 V_oc_single=V_oc_single_SPG,I_sc=I_sc_SPG,T_coef_V=T_coef_V_SPG,T_coef_I=T_coef_I_SPG,
#                 I_ph_ref=I_ph_ref_SPG,R_p=R_p_SPG,R_s=R_s_SPG,cell_length=cell_length_SPG,cell_width=cell_width_SPG,
#                 case_width=case_width_SPG,case_length=case_length_SPI,cell_diagonal=cell_diagonal_SP,cell_type='SunPower G')
#    SingleCellSunPowerH = Parameter(Area_Desired=0.0173,Voltage_Desired=0.69,N_s_guess=1,N_p_guess=1,#I_pv=1,
#                 V_oc_single=V_oc_single_SPH,I_sc=I_sc_SPH,T_coef_V=T_coef_V_SPH,T_coef_I=T_coef_I_SPH,
#                 I_ph_ref=I_ph_ref_SPH,R_p=R_p_SPH,R_s=R_s_SPH,cell_length=cell_length_SPH,cell_width=cell_width_SPH,
#                 case_width=case_width_SPH,case_length=case_length_SPH,cell_diagonal=cell_diagonal_SP,cell_type='SunPower H')
#    SingleCellSunPowerI = Parameter(Area_Desired=0.0173,Voltage_Desired=0.69,N_s_guess=1,N_p_guess=1,#I_pv=1,
#                 V_oc_single=V_oc_single_SPI,I_sc=I_sc_SPI,T_coef_V=T_coef_V_SPI,T_coef_I=T_coef_I_SPI,
#                 I_ph_ref=I_ph_ref_SPI,R_p=R_p_SPI,R_s=R_s_SPI,cell_length=cell_length_SPI,cell_width=cell_width_SPI,
#                 case_width=case_width_SPI,case_length=case_length_SPI,cell_diagonal=cell_diagonal_SP,cell_type='SunPower I')
#    SingleCellSunPowerJ = Parameter(Area_Desired=0.0173,Voltage_Desired=0.69,N_s_guess=1,N_p_guess=1,#I_pv=1,
#                 V_oc_single=V_oc_single_SPJ,I_sc=I_sc_SPJ,T_coef_V=T_coef_V_SPJ,T_coef_I=T_coef_I_SPJ,
#                 I_ph_ref=I_ph_ref_SPJ,R_p=R_p_SPJ,R_s=R_s_SPJ,cell_length=cell_length_SPJ,cell_width=cell_width_SPJ,
#                 case_width=case_width_SPJ,case_length=case_length_SPJ,cell_diagonal=cell_diagonal_SP,cell_type='SunPower J')
#    SingleCellAlta_hyb = Parameter(Area_Desired=0.0012,Voltage_Desired=1.1,N_s_guess=1,N_p_guess=1,
#                 V_oc_single=V_oc_single_Alta,I_sc=I_sc_Alta,T_coef_V=T_coef_V_Alta,T_coef_I=T_coef_I_Alta,a=a_Alta_hyb,
#                 I_ph_ref=I_ph_ref_Alta_hyb,R_p=R_p_Alta_hyb,R_s=R_s_Alta_hyb,cell_length=cell_length_Alta,
#                 cell_width=cell_width_Alta,case_width=case_width_Alta)
#    SingleCellSunPowerH_hyb = Parameter(Area_Desired=0.0173,Voltage_Desired=0.69,N_s_guess=1,N_p_guess=1,#I_pv=1,
#                 V_oc_single=V_oc_single_SPH,I_sc=I_sc_SPH,T_coef_V=T_coef_V_SPH,T_coef_I=T_coef_I_SPH,a=a_SPH_hyb,
#                 I_ph_ref=I_ph_ref_SPH_hyb,R_p=R_p_SPH_hyb,R_s=R_s_SPH_hyb,cell_length=cell_length_SPH,cell_width=cell_width_SPH,
#                 case_width=case_width_SPH,case_length=case_length_SPH,cell_diagonal=cell_diagonal_SP,cell_type='SunPower H')
#
#    SingleCellAltaOld.T = 25+273.15 # K
#    SingleCellAlta.T = 25+273.15 # K
#    SingleCellSunPowerG.T = 25+273.15 # K
#    SingleCellSunPowerH.T = 25+273.15 # K
#    SingleCellSunPowerI.T = 25+273.15 # K
#    SingleCellSunPowerJ.T = 25+273.15 # K
#    SingleCellAlta_hyb.T = 25+273.15 # K
#    SingleCellSunPowerH_hyb.T = 25+273.15 # K
#    voltages = np.linspace(0,1.15,1000)
#    CurrentsAltaOld = np.empty((len(voltages),4))
#    CurrentsAlta = np.empty((len(voltages),4))
#    CurrentsSunPowerG = np.empty((len(voltages),5))
#    CurrentsSunPowerH = np.empty((len(voltages),5))
#    CurrentsSunPowerI = np.empty((len(voltages),5))
#    CurrentsSunPowerJ = np.empty((len(voltages),5))
#    CurrentsAlta_hyb = np.empty((len(voltages),4))
#    CurrentsSunPowerH_hyb = np.empty((len(voltages),5))
#    for i in range(0,len(voltages)):
#        SingleCellAltaOld.V_pv = voltages[i]
#        SingleCellAlta.V_pv = voltages[i]
#        SingleCellSunPowerG.V_pv = voltages[i]
#        SingleCellSunPowerH.V_pv = voltages[i]
#        SingleCellSunPowerI.V_pv = voltages[i]
#        SingleCellSunPowerJ.V_pv = voltages[i]
#        SingleCellAlta_hyb.V_pv = voltages[i]
#        SingleCellSunPowerH_hyb.V_pv = voltages[i]
#
#        SingleCellAltaOld.G = 1000      # W/m^2
#        SingleCellAlta.G = 1000      # W/m^2
#        CurrentsAltaOld[i,0] = I_Single_String(SingleCellAltaOld)
#        CurrentsAlta[i,0] = I_Single_String(SingleCellAlta)
#        SingleCellSunPowerG.G = 1000  # W/m^2
#        SingleCellSunPowerH.G = 1000  # W/m^2
#        SingleCellSunPowerI.G = 1000  # W/m^2
#        SingleCellSunPowerJ.G = 1000  # W/m^2
#        CurrentsSunPowerG[i,0] = I_Single_String(SingleCellSunPowerG)
#        CurrentsSunPowerH[i,0] = I_Single_String(SingleCellSunPowerH)
#        CurrentsSunPowerI[i,0] = I_Single_String(SingleCellSunPowerI)
#        CurrentsSunPowerJ[i,0] = I_Single_String(SingleCellSunPowerJ)
#        SingleCellAlta_hyb.G = 1000  # W/m^2
#        SingleCellSunPowerH_hyb.G = 1000  # W/m^2
#        CurrentsAlta_hyb[i,0] = I_Single_String(SingleCellAlta_hyb,NREL=True)
#        CurrentsSunPowerH_hyb[i,0] = I_Single_String(SingleCellSunPowerH_hyb,NREL=True)
#        SingleCellSunPowerG.T = 50+273.15 # K
#        SingleCellSunPowerH.T = 50+273.15 # K
#        SingleCellSunPowerI.T = 50+273.15 # K
#        SingleCellSunPowerJ.T = 50+273.15 # K
#        SingleCellSunPowerH_hyb.T = 50+273.15 # K
#        CurrentsSunPowerG[i,4] = I_Single_String(SingleCellSunPowerG)
#        CurrentsSunPowerH[i,4] = I_Single_String(SingleCellSunPowerH)
#        CurrentsSunPowerI[i,4] = I_Single_String(SingleCellSunPowerI)
#        CurrentsSunPowerJ[i,4] = I_Single_String(SingleCellSunPowerJ)
#        CurrentsSunPowerH_hyb[i,4] = I_Single_String(SingleCellSunPowerH_hyb,NREL=True)
#        SingleCellSunPowerG.T = 25+273.15 # K
#        SingleCellSunPowerH.T = 25+273.15 # K
#        SingleCellSunPowerI.T = 25+273.15 # K
#        SingleCellSunPowerJ.T = 25+273.15 # K
#        SingleCellSunPowerH_hyb.T = 25+273.15 # K
#
#        SingleCellAltaOld.G = 500      # W/m^2
#        SingleCellAlta.G = 500      # W/m^2
#        CurrentsAltaOld[i,1] = I_Single_String(SingleCellAltaOld)
#        CurrentsAlta[i,1] = I_Single_String(SingleCellAlta)
#        SingleCellSunPowerG.G = 800  # W/m^2
#        SingleCellSunPowerH.G = 800  # W/m^2
#        SingleCellSunPowerI.G = 800  # W/m^2
#        SingleCellSunPowerJ.G = 800  # W/m^2
#        CurrentsSunPowerG[i,1] = I_Single_String(SingleCellSunPowerG)
#        CurrentsSunPowerH[i,1] = I_Single_String(SingleCellSunPowerH)
#        CurrentsSunPowerI[i,1] = I_Single_String(SingleCellSunPowerI)
#        CurrentsSunPowerJ[i,1] = I_Single_String(SingleCellSunPowerJ)
#        SingleCellAlta_hyb.G = 500  # W/m^2
#        SingleCellSunPowerH_hyb.G = 800  # W/m^2
#        CurrentsAlta_hyb[i,1] = I_Single_String(SingleCellAlta_hyb,NREL=True)
#        CurrentsSunPowerH_hyb[i,1] = I_Single_String(SingleCellSunPowerH_hyb,NREL=True)
#
#        SingleCellAltaOld.G = 200      # W/m^2
#        SingleCellAlta.G = 200      # W/m^2
#        CurrentsAltaOld[i,2] = I_Single_String(SingleCellAltaOld)
#        CurrentsAlta[i,2] = I_Single_String(SingleCellAlta)
#        SingleCellSunPowerG.G = 500  # W/m^2
#        SingleCellSunPowerH.G = 500  # W/m^2
#        SingleCellSunPowerI.G = 500  # W/m^2
#        SingleCellSunPowerJ.G = 500  # W/m^2
#        CurrentsSunPowerG[i,2] = I_Single_String(SingleCellSunPowerG)
#        CurrentsSunPowerH[i,2] = I_Single_String(SingleCellSunPowerH)
#        CurrentsSunPowerI[i,2] = I_Single_String(SingleCellSunPowerI)
#        CurrentsSunPowerJ[i,2] = I_Single_String(SingleCellSunPowerJ)
#        SingleCellAlta_hyb.G = 200  # W/m^2
#        SingleCellSunPowerH_hyb.G = 500  # W/m^2
#        CurrentsAlta_hyb[i,2] = I_Single_String(SingleCellAlta_hyb,NREL=True)
#        CurrentsSunPowerH_hyb[i,2] = I_Single_String(SingleCellSunPowerH_hyb,NREL=True)
#
#        SingleCellAltaOld.G = 100      # W/m^2
#        SingleCellAlta.G = 100      # W/m^2
#        CurrentsAltaOld[i,3] = I_Single_String(SingleCellAltaOld)
#        CurrentsAlta[i,3] = I_Single_String(SingleCellAlta)
#        SingleCellSunPowerG.G = 300  # W/m^2
#        SingleCellSunPowerH.G = 300  # W/m^2
#        SingleCellSunPowerI.G = 300  # W/m^2
#        SingleCellSunPowerJ.G = 300  # W/m^2
#        CurrentsSunPowerG[i,3] = I_Single_String(SingleCellSunPowerG)
#        CurrentsSunPowerH[i,3] = I_Single_String(SingleCellSunPowerH)
#        CurrentsSunPowerI[i,3] = I_Single_String(SingleCellSunPowerI)
#        CurrentsSunPowerJ[i,3] = I_Single_String(SingleCellSunPowerJ)
#        SingleCellAlta_hyb.G = 100  # W/m^2
#        SingleCellSunPowerH_hyb.G = 300  # W/m^2
#        CurrentsAlta_hyb[i,3] = I_Single_String(SingleCellAlta_hyb,NREL=True)
#        CurrentsSunPowerH_hyb[i,3] = I_Single_String(SingleCellSunPowerH_hyb,NREL=True)
#
##    for i in range(0,20):
##        plt.close(i)
##    plt.figure()
##    plt.plot(voltages,CurrentsSunPower)
##    plt.plot(voltages,CurrentsAlta)
##    plt.plot(voltages,CurrentsAltaOld,'--')
#    #np.amax(CurrentsSunPower)
#
#    plt.figure()
#    plt.plot(voltages, CurrentsAlta[:,0]*1000, 'b-', label=r'$1000\ W/m^2$')
#    plt.plot(voltages, CurrentsAlta[:,1]*1000, 'r-', label=r'$500\ W/m^2$')
#    plt.plot(voltages, CurrentsAlta[:,2]*1000, 'g-', label=r'$200\ W/m^2$')
#    plt.plot(voltages, CurrentsAlta[:,3]*1000, 'k-', label=r'$100\ W/m^2$')
#    plt.plot(voltages, CurrentsAltaOld[:,0]*1000, 'b--')
#    plt.plot(voltages, CurrentsAltaOld[:,1]*1000, 'r--')
#    plt.plot(voltages, CurrentsAltaOld[:,2]*1000, 'g--')
#    plt.plot(voltages, CurrentsAltaOld[:,3]*1000, 'k--')
#    plt.ylim(0,255)
#    plt.yticks([50,100,150,200,250])
#    plt.xlim(0,1.4)
#    plt.ylabel('Current (mA)')
#    plt.xlabel('Voltage (V)')
#    plt.title('Single Cell Solar Flux Variation (Parameters: New ––––  Old - - -)')
#    plt.legend(loc=1,prop={'size':9.5})
#    plt.grid()
#
#    plt.figure()
#    plt.plot(voltages, CurrentsAlta[:,0]*1000, 'b-', label=r'$1000\ W/m^2$')
#    plt.plot(voltages, CurrentsAlta[:,1]*1000, 'r-', label=r'$500\ W/m^2$')
#    plt.plot(voltages, CurrentsAlta[:,2]*1000, 'g-', label=r'$200\ W/m^2$')
#    plt.plot(voltages, CurrentsAlta[:,3]*1000, 'k-', label=r'$100\ W/m^2$')
#    plt.plot(voltages, CurrentsAlta_hyb[:,0]*1000, 'b--')
#    plt.plot(voltages, CurrentsAlta_hyb[:,1]*1000, 'r--')
#    plt.plot(voltages, CurrentsAlta_hyb[:,2]*1000, 'g--')
#    plt.plot(voltages, CurrentsAlta_hyb[:,3]*1000, 'k--')
#    plt.ylim(0,255)
#    plt.yticks([50,100,150,200,250])
#    plt.xlim(0,1.4)
#    plt.ylabel('Current (mA)')
#    plt.xlabel('Voltage (V)')
#    plt.title('Single Cell Solar Flux Variation (Models: Vika ––––  Hybrid - - -)')
#    plt.legend(loc=1,prop={'size':9.5})
#    plt.grid()
#
#    plt.figure()
#    plt.plot(voltages, CurrentsSunPowerG[:,0], 'k-', label=r'$1000\ W/m^2$')
#    plt.plot(voltages, CurrentsSunPowerG[:,4], 'b-', label=r'$1000\ W/m^2, 50°C$')
#    plt.plot(voltages, CurrentsSunPowerG[:,1], 'r-', label=r'$800\ W/m^2$')
#    plt.plot(voltages, CurrentsSunPowerG[:,2], 'g-', label=r'$500\ W/m^2$')
#    plt.plot(voltages, CurrentsSunPowerG[:,3], 'C4-', label=r'$300\ W/m^2$')
#    plt.plot(voltages, CurrentsSunPowerH[:,0], 'k--')
#    plt.plot(voltages, CurrentsSunPowerH[:,4], 'b--')
#    plt.plot(voltages, CurrentsSunPowerH[:,1], 'r--')
#    plt.plot(voltages, CurrentsSunPowerH[:,2], 'g--')
#    plt.plot(voltages, CurrentsSunPowerH[:,3], 'C4--')
#    plt.plot(voltages, CurrentsSunPowerI[:,0], 'k:')
#    plt.plot(voltages, CurrentsSunPowerI[:,4], 'b:')
#    plt.plot(voltages, CurrentsSunPowerI[:,1], 'r:')
#    plt.plot(voltages, CurrentsSunPowerI[:,2], 'g:')
#    plt.plot(voltages, CurrentsSunPowerI[:,3], 'C4:')
#    plt.plot(voltages, CurrentsSunPowerJ[:,0], 'k-.')
#    plt.plot(voltages, CurrentsSunPowerJ[:,4], 'b-.')
#    plt.plot(voltages, CurrentsSunPowerJ[:,1], 'r-.')
#    plt.plot(voltages, CurrentsSunPowerJ[:,2], 'g-.')
#    plt.plot(voltages, CurrentsSunPowerJ[:,3], 'C4-.')
##    plt.plot(voltages,np.ones(len(voltages))*(-1),'k-.',label='Bin J')
##    plt.plot(voltages,np.ones(len(voltages))*(-1),'k:',label='Bin I')
##    plt.plot(voltages,np.ones(len(voltages))*(-1),'k--',label='Bin H')
##    plt.plot(voltages,np.ones(len(voltages))*(-1),'k-',label='Bin G')
#    plt.ylim(0,7)
#    plt.xlim(0,0.8)
#    plt.xticks([0,0.2,0.4,0.6,0.8])
#    plt.ylabel('Current (A)')
#    plt.xlabel('Voltage (V)')
#    plt.title('Single Cell Solar Flux Variation (SunPower)')
#    plt.legend(loc='lower left',prop={'size':9.5},ncol=2)
#    plt.grid()
#
#    plt.figure()
#    plt.plot(voltages, CurrentsSunPowerH[:,0], 'k-', label=r'$1000\ W/m^2$')
#    plt.plot(voltages, CurrentsSunPowerH[:,4], 'b-', label=r'$1000\ W/m^2, 50°C$')
#    plt.plot(voltages, CurrentsSunPowerH[:,1], 'r-', label=r'$800\ W/m^2$')
#    plt.plot(voltages, CurrentsSunPowerH[:,2], 'g-', label=r'$500\ W/m^2$')
#    plt.plot(voltages, CurrentsSunPowerH[:,3], 'C4-', label=r'$300\ W/m^2$')
#    plt.plot(voltages, CurrentsSunPowerH_hyb[:,0], 'k--')
#    plt.plot(voltages, CurrentsSunPowerH_hyb[:,4], 'b--')
#    plt.plot(voltages, CurrentsSunPowerH_hyb[:,1], 'r--')
#    plt.plot(voltages, CurrentsSunPowerH_hyb[:,2], 'g--')
#    plt.plot(voltages, CurrentsSunPowerH_hyb[:,3], 'C4--')
#    plt.ylim(0,7)
#    plt.xlim(0,0.8)
#    plt.xticks([0,0.2,0.4,0.6,0.8])
#    plt.ylabel('Current (A)')
#    plt.xlabel('Voltage (V)')
#    plt.title('Single Cell Solar Flux Variation (Models: Vika ––––  Hybrid - - -)')
#    plt.legend(loc='lower left',prop={'size':9.5},ncol=2)
#    plt.grid()
#
#    plt.figure()
#    plt.plot(voltages, CurrentsSunPowerH[:,0], 'k--', label=r'$1000\ W/m^2$')
#    plt.plot(voltages, CurrentsSunPowerH[:,4], 'b--', label=r'$1000\ W/m^2, 50°C$')
#    plt.plot(voltages, CurrentsSunPowerH[:,1], 'r--', label=r'$800\ W/m^2$')
#    plt.plot(voltages, CurrentsSunPowerH[:,2], 'g--', label=r'$500\ W/m^2$')
#    plt.plot(voltages, CurrentsSunPowerH[:,3], 'C4--', label=r'$300\ W/m^2$')
#    plt.plot(voltages, CurrentsAlta[:,0], 'k-')
#    plt.plot(voltages, CurrentsAlta[:,1], 'g-')
#    plt.plot(voltages, CurrentsAlta[:,2], '-', color='pink',label=r'$200\ W/m^2$')
#    plt.plot(voltages, CurrentsAlta[:,3], '-', color='orange',label=r'$100\ W/m^2$')
#    plt.ylim(0,7)
#    plt.xlim(0,1.2)
#    plt.ylabel('Current (A)')
#    plt.xlabel('Voltage (V)')
#    plt.title('Single Cell Solar Flux Variation (Cell Type: Alta ––––  SunPower - - -)')
#    plt.legend(loc='upper right',prop={'size':9.5},ncol=1)
#    plt.grid()
#
#    # Efficiency vs. Flux Plots
#    T = np.array([25,0,-25,-50,-75]) + 273.15  # (K)
#    nlines = len(T)
#    flux = np.linspace(0,1200,500) # (W/m^2)
#    AltaOld_efficiencies = np.zeros((len(flux),nlines))
#    Alta_efficiencies = np.zeros((len(flux),nlines))
#    SunPower_efficienciesG = np.zeros((len(flux),nlines))
#    SunPower_efficienciesH = np.zeros((len(flux),nlines))
#    SunPower_efficienciesI = np.zeros((len(flux),nlines))
#    SunPower_efficienciesJ = np.zeros((len(flux),nlines))
#    Alta_efficiencies_hyb = np.zeros((len(flux),nlines))
#    SunPower_efficienciesH_hyb = np.zeros((len(flux),nlines))
#
#    for i in range(0,len(flux)):
#        SingleCellAltaOld.G = flux[i]
#        SingleCellAlta.G = flux[i]
#        SingleCellSunPowerG.G = flux[i]
#        SingleCellSunPowerH.G = flux[i]
#        SingleCellSunPowerI.G = flux[i]
#        SingleCellSunPowerJ.G = flux[i]
#        SingleCellAlta_hyb.G = flux[i]
#        SingleCellSunPowerH_hyb.G = flux[i]
#        for j in range(0,nlines):
#            SingleCellAltaOld.T = T[j]
#            SingleCellAlta.T = T[j]
#            SingleCellSunPowerG.T = T[j]
#            SingleCellSunPowerH.T = T[j]
#            SingleCellSunPowerI.T = T[j]
#            SingleCellSunPowerJ.T = T[j]
#            SingleCellAlta_hyb.T = T[j]
#            SingleCellSunPowerH_hyb.T = T[j]
#            AltaOld_efficiencies[i][j] = SolarPowerMPP([SingleCellAltaOld])[8]
#            Alta_efficiencies[i][j] = SolarPowerMPP([SingleCellAlta])[8]
#            SunPower_efficienciesG[i][j] = SolarPowerMPP([SingleCellSunPowerG])[8]
#            SunPower_efficienciesH[i][j] = SolarPowerMPP([SingleCellSunPowerH])[8]
#            SunPower_efficienciesI[i][j] = SolarPowerMPP([SingleCellSunPowerI])[8]
#            SunPower_efficienciesJ[i][j] = SolarPowerMPP([SingleCellSunPowerJ])[8]
#            Alta_efficiencies_hyb[i][j] = SolarPowerMPP([SingleCellAlta_hyb],NREL=True)[8]
#            SunPower_efficienciesH_hyb[i][j] = SolarPowerMPP([SingleCellSunPowerH_hyb],NREL=True)[8]
#    for i in range(0,len(flux)-1):
#        if flux[i+1] > 1000 and flux[i] < 1000:
#            Flux1000_25_AltaOld = AltaOld_efficiencies[i,0]
#            Flux1000_25_Alta = Alta_efficiencies[i,0]
#            Flux1000_25_SPG = SunPower_efficienciesG[i,0]
#            Flux1000_25_SPH = SunPower_efficienciesH[i,0]
#            Flux1000_25_SPI = SunPower_efficienciesI[i,0]
#            Flux1000_25_SPJ = SunPower_efficienciesJ[i,0]
#            Flux1000_25_Alta_hyb = Alta_efficiencies_hyb[i,0]
#            Flux1000_25_SPH_hyb = SunPower_efficienciesH_hyb[i,0]
#            break
#
#    plt.figure()
#    plt.title("Alta Devices Efficiencies vs. Flux (Parameters: New ––––  Old - - -)")
#    plt.plot(flux,Alta_efficiencies[:,0],'g-',label=str(int(T[0]-273.15))+" °C")
#    plt.plot(flux,Alta_efficiencies[:,1],'b-',label=str(int(T[1]-273.15))+" °C")
#    plt.plot(flux,Alta_efficiencies[:,2],'r-',label=str(int(T[2]-273.15))+" °C")
#    plt.plot(flux,Alta_efficiencies[:,3],'c-',label=str(int(T[3]-273.15))+" °C")
#    plt.plot(flux,Alta_efficiencies[:,4],'m-',label=str(int(T[4]-273.15))+" °C")
#    plt.plot(flux,AltaOld_efficiencies[:,0],'g--')
#    plt.plot(flux,AltaOld_efficiencies[:,1],'b--')
#    plt.plot(flux,AltaOld_efficiencies[:,2],'r--')
#    plt.plot(flux,AltaOld_efficiencies[:,3],'c--')
#    plt.plot(flux,AltaOld_efficiencies[:,4],'m--')
#    plt.plot(1000,Flux1000_25_AltaOld,'go',alpha=0.9)
#    plt.plot(1000,Flux1000_25_Alta,'go')
#    plt.text(200,0.16, 'Old: %.2f' % (Flux1000_25_AltaOld*100) +' %',color='g',ha='left')
#    plt.text(200,0.153,'New: %.2f' % (Flux1000_25_Alta*100) +' %',color='g',ha='left')
#    plt.xlim([0,flux[-1]])
#    plt.ylim([0.15,0.30])
#    plt.ylabel("Efficiency")
#    plt.xlabel("Flux (W/m^2)")
#    plt.legend(loc=4, ncol = 2)
#
#    plt.figure()
#    plt.title("Alta Devices Efficiencies vs. Flux (Models: Vika ––––  Hybrid - - -)")
#    plt.plot(flux,Alta_efficiencies[:,0],'g-',label=str(int(T[0]-273.15))+" °C")
#    plt.plot(flux,Alta_efficiencies[:,1],'b-',label=str(int(T[1]-273.15))+" °C")
#    plt.plot(flux,Alta_efficiencies[:,2],'r-',label=str(int(T[2]-273.15))+" °C")
#    plt.plot(flux,Alta_efficiencies[:,3],'c-',label=str(int(T[3]-273.15))+" °C")
#    plt.plot(flux,Alta_efficiencies[:,4],'m-',label=str(int(T[4]-273.15))+" °C")
#    plt.plot(flux,Alta_efficiencies_hyb[:,0],'g--')
#    plt.plot(flux,Alta_efficiencies_hyb[:,1],'b--')
#    plt.plot(flux,Alta_efficiencies_hyb[:,2],'r--')
#    plt.plot(flux,Alta_efficiencies_hyb[:,3],'c--')
#    plt.plot(flux,Alta_efficiencies_hyb[:,4],'m--')
#    plt.plot(1000,Flux1000_25_Alta_hyb,'go',alpha=0.9)
#    plt.plot(1000,Flux1000_25_Alta,'go')
#    plt.text(200,0.16, 'Hybrid: %.2f' % (Flux1000_25_Alta_hyb*100) +' %',color='g',ha='left')
#    plt.text(200,0.153,'  Vika: %.2f' % (Flux1000_25_Alta*100) +' %',color='g',ha='left')
#    plt.xlim([0,flux[-1]])
#    plt.ylim([0.15,0.30])
#    plt.ylabel("Efficiency")
#    plt.xlabel("Flux (W/m^2)")
#    plt.legend(loc=4, ncol = 2)
#
#    plt.figure()
#    plt.title("Efficiencies vs. Flux (Cell Type: Alta ––––  SunPower - - -)")
#    plt.plot(flux,Alta_efficiencies[:,0],'g-',label=str(int(T[0]-273.15))+" °C")
#    plt.plot(flux,Alta_efficiencies[:,1],'b-',label=str(int(T[1]-273.15))+" °C")
#    plt.plot(flux,Alta_efficiencies[:,2],'r-',label=str(int(T[2]-273.15))+" °C")
#    plt.plot(flux,Alta_efficiencies[:,3],'c-',label=str(int(T[3]-273.15))+" °C")
#    plt.plot(flux,Alta_efficiencies[:,4],'m-',label=str(int(T[4]-273.15))+" °C")
#    plt.plot(flux,SunPower_efficienciesH[:,0],'g--')
#    plt.plot(flux,SunPower_efficienciesH[:,1],'b--')
#    plt.plot(flux,SunPower_efficienciesH[:,2],'r--')
#    plt.plot(flux,SunPower_efficienciesH[:,3],'c--')
#    plt.plot(flux,SunPower_efficienciesH[:,4],'m--')
#    plt.plot(1000,Flux1000_25_Alta,'go',alpha=0.9)
#    plt.plot(1000,Flux1000_25_SPH,'go')
#    plt.text(200,0.16, 'Alta: %.2f' % (Flux1000_25_Alta*100) +' %',color='g',ha='left')
#    plt.text(200,0.153,'SunPower: %.2f' % (Flux1000_25_SPH*100) +' %',color='g',ha='left')
#    plt.xlim([0,flux[-1]])
#    plt.ylim([0.15,0.30])
#    plt.ylabel("Efficiency")
#    plt.xlabel("Flux (W/m^2)")
#    plt.legend(loc=4, ncol = 2)
#
#    plt.figure()
#    plt.title("SunPower Efficiencies vs. Flux")
#    plt.plot(flux,SunPower_efficienciesG[:,4],'m-',label=str(int(T[4]-273.15))+" °C")
#    plt.plot(flux,SunPower_efficienciesG[:,3],'c-',label=str(int(T[3]-273.15))+" °C")
#    plt.plot(flux,SunPower_efficienciesG[:,2],'r-',label=str(int(T[2]-273.15))+" °C")
#    plt.plot(flux,SunPower_efficienciesG[:,1],'b-',label=str(int(T[1]-273.15))+" °C")
#    plt.plot(flux,SunPower_efficienciesG[:,0],'g-',label=str(int(T[0]-273.15))+" °C")
#    plt.plot(flux,SunPower_efficienciesH[:,0],'g--')
#    plt.plot(flux,SunPower_efficienciesH[:,1],'b--')
#    plt.plot(flux,SunPower_efficienciesH[:,2],'r--')
#    plt.plot(flux,SunPower_efficienciesH[:,3],'c--')
#    plt.plot(flux,SunPower_efficienciesH[:,4],'m--')
#    plt.plot(flux,SunPower_efficienciesI[:,0],'g:')
#    plt.plot(flux,SunPower_efficienciesI[:,1],'b:')
#    plt.plot(flux,SunPower_efficienciesI[:,2],'r:')
#    plt.plot(flux,SunPower_efficienciesI[:,3],'c:')
#    plt.plot(flux,SunPower_efficienciesI[:,4],'m:')
#    plt.plot(flux,SunPower_efficienciesJ[:,0],'g-.')
#    plt.plot(flux,SunPower_efficienciesJ[:,1],'b-.')
#    plt.plot(flux,SunPower_efficienciesJ[:,2],'r-.')
#    plt.plot(flux,SunPower_efficienciesJ[:,3],'c-.')
#    plt.plot(flux,SunPower_efficienciesJ[:,4],'m-.')
#    plt.plot(flux,np.ones(len(flux)),'k-.',label='Bin J')
#    plt.plot(flux,np.ones(len(flux)),'k:',label='Bin I')
#    plt.plot(flux,np.ones(len(flux)),'k--',label='Bin H')
#    plt.plot(flux,np.ones(len(flux)),'k-',label='Bin G')
#    plt.plot(1000,Flux1000_25_SPG,'go')
#    plt.plot(1000,Flux1000_25_SPH,'go')
#    plt.plot(1000,Flux1000_25_SPI,'go')
#    plt.plot(1000,Flux1000_25_SPJ,'go')
#    plt.text(200,0.174,'J: %.2f' % (Flux1000_25_SPJ*100) +' %',color='g',ha='left')
#    plt.text(200,0.167,'I: %.2f' % (Flux1000_25_SPI*100) +' %',color='g',ha='left')
#    plt.text(200,0.16,'H: %.2f' % (Flux1000_25_SPH*100) +' %',color='g',ha='left')
#    plt.text(200,0.153,'G: %.2f' % (Flux1000_25_SPG*100) +' %',color='g',ha='left')
#    plt.xlim([0,flux[-1]])
#    plt.ylim([0.15,0.30])
#    plt.ylabel("Efficiency")
#    plt.xlabel("Flux (W/m^2)")
#    plt.legend(loc=4, ncol = 2)
#
#    plt.figure()
#    plt.title("SunPower Efficiencies vs. Flux (Models: Vika ––––  Hybrid - - -)")
#    plt.plot(flux,SunPower_efficienciesH[:,4],'m-',label=str(int(T[4]-273.15))+" °C")
#    plt.plot(flux,SunPower_efficienciesH[:,3],'c-',label=str(int(T[3]-273.15))+" °C")
#    plt.plot(flux,SunPower_efficienciesH[:,2],'r-',label=str(int(T[2]-273.15))+" °C")
#    plt.plot(flux,SunPower_efficienciesH[:,1],'b-',label=str(int(T[1]-273.15))+" °C")
#    plt.plot(flux,SunPower_efficienciesH[:,0],'g-',label=str(int(T[0]-273.15))+" °C")
#    plt.plot(flux,SunPower_efficienciesH_hyb[:,0],'g--')
#    plt.plot(flux,SunPower_efficienciesH_hyb[:,1],'b--')
#    plt.plot(flux,SunPower_efficienciesH_hyb[:,2],'r--')
#    plt.plot(flux,SunPower_efficienciesH_hyb[:,3],'c--')
#    plt.plot(flux,SunPower_efficienciesH_hyb[:,4],'m--')
#    plt.plot(1000,Flux1000_25_SPH_hyb,'go',alpha=0.9)
#    plt.plot(1000,Flux1000_25_SPH,'go')
#    plt.text(200,0.16,'   Vika: %.2f' % (Flux1000_25_SPH*100) +' %',color='g',ha='left')
#    plt.text(200,0.153,'Hybrid: %.2f' % (Flux1000_25_SPH_hyb*100) +' %',color='g',ha='left')
#    plt.xlim([0,flux[-1]])
#    plt.ylim([0.15,0.30])
#    plt.ylabel("Efficiency")
#    plt.xlabel("Flux (W/m^2)")
#    plt.legend(loc=4, ncol = 2)
#
##    test = Parameter(Area_Desired=60.0,Voltage_Desired=100,N_s_guess=50,N_p_guess=50,I_pv=1,
##                 V_oc_single=V_oc_single_SPH,I_sc=I_sc_SPH,T_coef_V=T_coef_V_SPH,T_coef_I=T_coef_I_SPH,
##                 I_ph_ref=I_ph_ref_SPH,R_p=R_p_SPH,R_s=R_s_SPH,cell_length=cell_length_SPH,cell_width=cell_width_SPH,
##                 case_width=case_width_SPH,case_length=case_length_SPH,cell_type='SunPower H')
##    print (test.N_s,test.SolarPanelArea,test.SolarVoltageOpenCircuit,test.N_p)
##
##    test2 = Parameter(Area_Desired=60.0,Voltage_Desired=100,N_s_guess=1,N_p_guess=1,
##                 V_oc_single=V_oc_single_Alta,I_sc=I_sc_Alta,T_coef_V=T_coef_V_Alta,T_coef_I=T_coef_I_Alta,
##                 I_ph_ref=I_ph_ref_Alta,R_p=R_p_Alta,R_s=R_s_Alta,cell_length=cell_length_Alta,cell_width=cell_width_Alta,case_width=case_width_Alta)
##    print (test2.N_s,test2.SolarPanelArea,test2.SolarVoltageOpenCircuit,test2.N_p)
#    # For an entire array
#        # SectionsPerWing = 16, VoltageArrayOC = 80 is a good fit for SunPower cells
#
#    import os
#    file_path = os.getcwd()
#    os.chdir('./Airfoil_Sections_Local_Flux')
#    from airfoil_sections import calc_xy_coordinates
#    os.chdir(file_path)
#    repo_path = file_path.replace('Simulation','')
#
#    SectionsPerWing = 16
#    VoltageArrayOC = 25 # (V) [open circuit voltage of the entire array]
#    [NewXCoordinates, NewYCoordinates, XMidpoints, YMidpoints, desired_len] = calc_xy_coordinates(NumOfSections3=SectionsPerWing,RepositoryPath3=repo_path)
#
#    wing_span = 42.4  # (m)
#    sweep_angle = 20 # (degrees)
#    wing_length =  wing_span/2/np.cos(np.radians(sweep_angle)) # (m)
#
#    lengths = np.empty(SectionsPerWing)
#    Areas = np.empty(SectionsPerWing*2)
#    for i in range(0,SectionsPerWing*2):
#        if i < SectionsPerWing:
#            delta_x = NewXCoordinates[i+1]-NewXCoordinates[i]
#            delta_y = NewYCoordinates[i+1]-NewYCoordinates[i]
#            lengths[i] = np.sqrt(delta_x**2 + delta_y**2)
#            Areas[i] = lengths[i]*wing_length*np.cos(np.radians(sweep_angle))  # (m^2) [shaped like a parallelogram]
#        else:
#            Areas[i] = Areas[i-SectionsPerWing]
#    total_area = sum(Areas)
#
#    SolarSectionsAltaOld = []
#    SolarSectionsAlta = []
#    SolarSectionsSunPowerG = []
#    SolarSectionsSunPowerH = []
#    SolarSectionsSunPowerI = []
#    SolarSectionsSunPowerJ = []
#    SolarSectionsAlta_hyb = []
#    SolarSectionsSunPowerH_hyb = []
#    if len(Areas) == SectionsPerWing*2:
#        for i in range(0,SectionsPerWing*2):
#            SolarSectionsAltaOld.append(Parameter(T=25+273.15,Area_Desired=Areas[i],Voltage_Desired=VoltageArrayOC,
#                         V_oc_single=V_oc_single_Alta,I_sc=I_sc_Alta,T_coef_V=T_coef_V_Alta,T_coef_I=T_coef_I_Alta,
#                         I_ph_ref=I_ph_ref_AltaOld,R_p=R_p_AltaOld,R_s=R_s_AltaOld,cell_length=cell_length_Alta,
#                         cell_width=cell_width_Alta,case_width=case_width_Alta))
#            SolarSectionsAlta.append(Parameter(T=25+273.15,Area_Desired=Areas[i],Voltage_Desired=VoltageArrayOC,
#                         V_oc_single=V_oc_single_Alta,I_sc=I_sc_Alta,T_coef_V=T_coef_V_Alta,T_coef_I=T_coef_I_Alta,
#                         I_ph_ref=I_ph_ref_Alta,R_p=R_p_Alta,R_s=R_s_Alta,cell_length=cell_length_Alta,
#                         cell_width=cell_width_Alta,case_width=case_width_Alta))
#            SolarSectionsSunPowerG.append(Parameter(T=25+273.15,Area_Desired=Areas[i],Voltage_Desired=VoltageArrayOC,
#                         V_oc_single=V_oc_single_SPG,I_sc=I_sc_SPG,T_coef_V=T_coef_V_SPG,T_coef_I=T_coef_I_SPG,
#                         I_ph_ref=I_ph_ref_SPG,R_p=R_p_SPG,R_s=R_s_SPG,cell_length=cell_length_SPG,cell_width=cell_width_SPG,
#                         case_width=case_width_SPG,case_length=case_length_SPI,cell_diagonal=cell_diagonal_SP,cell_type='SunPower G'))
#            SolarSectionsSunPowerH.append(Parameter(T=25+273.15,Area_Desired=Areas[i],Voltage_Desired=VoltageArrayOC,
#                         V_oc_single=V_oc_single_SPH,I_sc=I_sc_SPH,T_coef_V=T_coef_V_SPH,T_coef_I=T_coef_I_SPH,
#                         I_ph_ref=I_ph_ref_SPH,R_p=R_p_SPH,R_s=R_s_SPH,cell_length=cell_length_SPH,cell_width=cell_width_SPH,
#                         case_width=case_width_SPH,case_length=case_length_SPH,cell_diagonal=cell_diagonal_SP,cell_type='SunPower H'))
#            SolarSectionsSunPowerI.append(Parameter(T=25+273.15,Area_Desired=Areas[i],Voltage_Desired=VoltageArrayOC,
#                         V_oc_single=V_oc_single_SPI,I_sc=I_sc_SPI,T_coef_V=T_coef_V_SPI,T_coef_I=T_coef_I_SPI,
#                         I_ph_ref=I_ph_ref_SPI,R_p=R_p_SPI,R_s=R_s_SPI,cell_length=cell_length_SPI,cell_width=cell_width_SPI,
#                         case_width=case_width_SPI,case_length=case_length_SPI,cell_diagonal=cell_diagonal_SP,cell_type='SunPower I'))
#            SolarSectionsSunPowerJ.append(Parameter(T=25+273.15,Area_Desired=Areas[i],Voltage_Desired=VoltageArrayOC,
#                         V_oc_single=V_oc_single_SPJ,I_sc=I_sc_SPJ,T_coef_V=T_coef_V_SPJ,T_coef_I=T_coef_I_SPJ,
#                         I_ph_ref=I_ph_ref_SPJ,R_p=R_p_SPJ,R_s=R_s_SPJ,cell_length=cell_length_SPJ,cell_width=cell_width_SPJ,
#                         case_width=case_width_SPJ,case_length=case_length_SPJ,cell_diagonal=cell_diagonal_SP,cell_type='SunPower J'))
#            SolarSectionsAlta_hyb.append(Parameter(T=25+273.15,Area_Desired=Areas[i],Voltage_Desired=VoltageArrayOC,
#                         V_oc_single=V_oc_single_Alta,I_sc=I_sc_Alta,T_coef_V=T_coef_V_Alta,T_coef_I=T_coef_I_Alta,a=a_Alta_hyb,
#                         I_ph_ref=I_ph_ref_Alta_hyb,R_p=R_p_Alta_hyb,R_s=R_s_Alta_hyb,cell_length=cell_length_Alta,
#                         cell_width=cell_width_Alta,case_width=case_width_Alta))
#            SolarSectionsSunPowerH_hyb.append(Parameter(T=25+273.15,Area_Desired=Areas[i],Voltage_Desired=VoltageArrayOC,
#                         V_oc_single=V_oc_single_SPH,I_sc=I_sc_SPH,T_coef_V=T_coef_V_SPH,T_coef_I=T_coef_I_SPH,a=a_SPH_hyb,
#                         I_ph_ref=I_ph_ref_SPH_hyb,R_p=R_p_SPH_hyb,R_s=R_s_SPH_hyb,cell_length=cell_length_SPH,cell_width=cell_width_SPH,
#                         case_width=case_width_SPH,case_length=case_length_SPH,cell_diagonal=cell_diagonal_SP,cell_type='SunPower H'))
#    else:
#        print ("\n\t\tAreas, Temperatures and SolarPanelSections are not of equal length")
#
#    total_area_panels_Alta = 0
#    total_area_cells_Alta = 0
#    total_area_panels_SP= 0
#    total_area_cells_SP = 0
#    for i in range(0,SectionsPerWing*2):
#        total_area_panels_Alta += SolarSectionsAlta[i].SolarPanelArea
#        total_area_cells_Alta += SolarSectionsAlta[i].SolarCellArea
#        total_area_panels_SP += SolarSectionsSunPowerH[i].SolarPanelArea
#        total_area_cells_SP += SolarSectionsSunPowerH[i].SolarCellArea
#    print ('\tArea Per Section: %.3f (m^2)' % Areas[0])
#    print ('\tAlta:')
#    print ('\t\tSolar Cell Area: ' + str(total_area_cells_Alta) + ' (m^2)')
#    print ('\t\tSolar Panel Area: ' + str(total_area_panels_Alta) + ' (m^2)')
#    print ('\t\tPercentage of Wing Area Used: ' + ('%.2f' % (total_area_panels_Alta/total_area*100)) + '%' +' (' + ('%.2f' % total_area_panels_Alta) + '/' + ('%.2f' %total_area) + ')')
#    print ('\t\tStrings In Parallel Per Section: ' + str(SolarSectionsAlta[0].N_p))
#    print ('\t\tArea Per String of Cells: ' + '%.3f (m^2)' % (SolarSectionsAlta[0].SolarPanelArea/SolarSectionsAlta[0].N_p))
#    print ('\tSunPower:')
#    print ('\t\tSolar Cell Area: ' + str(total_area_cells_SP) + ' (m^2)')
#    print ('\t\tSolar Panel Area: ' + str(total_area_panels_SP) + ' (m^2)')
#    print ('\t\tPercentage of Wing Area Used: ' + ('%.2f' % (total_area_panels_SP/total_area*100)) + '%' +' (' + ('%.2f' % total_area_panels_SP) + '/' + ('%.2f' %total_area) + ')')
#    print ('\t\tStrings In Parallel Per Section: ' + str(SolarSectionsSunPowerG[0].N_p))
#    print ('\t\tArea Per String of Cells: ' + '%.3f (m^2)' % (SolarSectionsSunPowerG[0].SolarPanelArea))
#
#    ## IV Curve for entire array
#    voltages = np.linspace(0,VoltageArrayOC*1.2,1000)
#    CurrentsArrayAltaOld = np.empty((len(voltages),5))
#    CurrentsArrayAlta = np.empty((len(voltages),5))
#    CurrentsArraySunPowerG = np.empty((len(voltages),5))
#    CurrentsArraySunPowerH = np.empty((len(voltages),5))
#    CurrentsArraySunPowerI = np.empty((len(voltages),5))
#    CurrentsArraySunPowerJ = np.empty((len(voltages),5))
#    CurrentsArrayAlta_hyb = np.empty((len(voltages),5))
#    CurrentsArraySunPowerH_hyb = np.empty((len(voltages),5))
#    for i in range(0,len(voltages)):
#        for k in range(0,SectionsPerWing*2):
#            SolarSectionsAltaOld[k].G = 1000      # W/m^2
#            SolarSectionsAlta[k].G = 1000      # W/m^2
#            SolarSectionsSunPowerG[k].G = 1000  # W/m^2
#            SolarSectionsSunPowerH[k].G = 1000  # W/m^2
#            SolarSectionsSunPowerI[k].G = 1000  # W/m^2
#            SolarSectionsSunPowerJ[k].G = 1000  # W/m^2
#            SolarSectionsAlta_hyb[k].G = 1000      # W/m^2
#            SolarSectionsSunPowerH_hyb[k].G = 1000  # W/m^2
#
#        CurrentsArrayAltaOld[i,0] = SolarPowerWith_DC_DC_Conversion(voltages[i],1.0,1.0,SolarSectionsAltaOld)[2]
#        CurrentsArrayAlta[i,0] = SolarPowerWith_DC_DC_Conversion(voltages[i],1.0,1.0,SolarSectionsAlta)[2]
#        CurrentsArraySunPowerG[i,0] = SolarPowerWith_DC_DC_Conversion(voltages[i],1.0,1.0,SolarSectionsSunPowerG)[2]
#        CurrentsArraySunPowerH[i,0] = SolarPowerWith_DC_DC_Conversion(voltages[i],1.0,1.0,SolarSectionsSunPowerH)[2]
#        CurrentsArraySunPowerI[i,0] = SolarPowerWith_DC_DC_Conversion(voltages[i],1.0,1.0,SolarSectionsSunPowerI)[2]
#        CurrentsArraySunPowerJ[i,0] = SolarPowerWith_DC_DC_Conversion(voltages[i],1.0,1.0,SolarSectionsSunPowerJ)[2]
#        CurrentsArrayAlta_hyb[i,0] = SolarPowerWith_DC_DC_Conversion(voltages[i],1.0,1.0,SolarSectionsAlta_hyb,NREL=True)[2]
#        CurrentsArraySunPowerH_hyb[i,0] = SolarPowerWith_DC_DC_Conversion(voltages[i],1.0,1.0,SolarSectionsSunPowerH_hyb,NREL=True)[2]
#        for k in range(0,SectionsPerWing*2):
#            SolarSectionsAltaOld[k].T = 50+273.15 # K
#            SolarSectionsAlta[k].T = 50+273.15 # K
#            SolarSectionsSunPowerG[k].T = 50+273.15 # K
#            SolarSectionsSunPowerH[k].T = 50+273.15 # K
#            SolarSectionsSunPowerI[k].T = 50+273.15 # K
#            SolarSectionsSunPowerJ[k].T = 50+273.15 # K
#            SolarSectionsAlta_hyb[k].T = 50+273.15 # K
#            SolarSectionsSunPowerH_hyb[k].T = 50+273.15 # K
#        CurrentsArrayAltaOld[i,4] = SolarPowerWith_DC_DC_Conversion(voltages[i],1.0,1.0,SolarSectionsAltaOld)[2]
#        CurrentsArrayAlta[i,4] = SolarPowerWith_DC_DC_Conversion(voltages[i],1.0,1.0,SolarSectionsAlta)[2]
#        CurrentsArraySunPowerG[i,4] = SolarPowerWith_DC_DC_Conversion(voltages[i],1.0,1.0,SolarSectionsSunPowerG)[2]
#        CurrentsArraySunPowerH[i,4] = SolarPowerWith_DC_DC_Conversion(voltages[i],1.0,1.0,SolarSectionsSunPowerH)[2]
#        CurrentsArraySunPowerI[i,4] = SolarPowerWith_DC_DC_Conversion(voltages[i],1.0,1.0,SolarSectionsSunPowerI)[2]
#        CurrentsArraySunPowerJ[i,4] = SolarPowerWith_DC_DC_Conversion(voltages[i],1.0,1.0,SolarSectionsSunPowerJ)[2]
#        CurrentsArrayAlta_hyb[i,4] = SolarPowerWith_DC_DC_Conversion(voltages[i],1.0,1.0,SolarSectionsAlta_hyb,NREL=True)[2]
#        CurrentsArraySunPowerH_hyb[i,4] = SolarPowerWith_DC_DC_Conversion(voltages[i],1.0,1.0,SolarSectionsSunPowerH_hyb,NREL=True)[2]
#        for k in range(0,SectionsPerWing*2):
#            SolarSectionsAltaOld[k].T = 25+273.15 # K
#            SolarSectionsAlta[k].T = 25+273.15 # K
#            SolarSectionsSunPowerG[k].T = 25+273.15 # K
#            SolarSectionsSunPowerH[k].T = 25+273.15 # K
#            SolarSectionsSunPowerI[k].T = 25+273.15 # K
#            SolarSectionsSunPowerJ[k].T = 25+273.15 # K
#            SolarSectionsAlta_hyb[k].T = 25+273.15 # K
#            SolarSectionsSunPowerH_hyb[k].T = 25+273.15 # K
#
#        for k in range(0,SectionsPerWing*2):
#            SolarSectionsAltaOld[k].G = 800      # W/m^2
#            SolarSectionsAlta[k].G = 800      # W/m^2
#            SolarSectionsSunPowerG[k].G = 800  # W/m^2
#            SolarSectionsSunPowerH[k].G = 800  # W/m^2
#            SolarSectionsSunPowerI[k].G = 800  # W/m^2
#            SolarSectionsSunPowerJ[k].G = 800  # W/m^2
#            SolarSectionsAlta_hyb[k].G = 800      # W/m^2
#            SolarSectionsSunPowerH_hyb[k].G = 800  # W/m^2
#        CurrentsArrayAltaOld[i,1] = SolarPowerWith_DC_DC_Conversion(voltages[i],1.0,1.0,SolarSectionsAltaOld)[2]
#        CurrentsArrayAlta[i,1] = SolarPowerWith_DC_DC_Conversion(voltages[i],1.0,1.0,SolarSectionsAlta)[2]
#        CurrentsArraySunPowerG[i,1] = SolarPowerWith_DC_DC_Conversion(voltages[i],1.0,1.0,SolarSectionsSunPowerG)[2]
#        CurrentsArraySunPowerH[i,1] = SolarPowerWith_DC_DC_Conversion(voltages[i],1.0,1.0,SolarSectionsSunPowerH)[2]
#        CurrentsArraySunPowerI[i,1] = SolarPowerWith_DC_DC_Conversion(voltages[i],1.0,1.0,SolarSectionsSunPowerI)[2]
#        CurrentsArraySunPowerJ[i,1] = SolarPowerWith_DC_DC_Conversion(voltages[i],1.0,1.0,SolarSectionsSunPowerJ)[2]
#        CurrentsArrayAlta_hyb[i,1] = SolarPowerWith_DC_DC_Conversion(voltages[i],1.0,1.0,SolarSectionsAlta_hyb,NREL=True)[2]
#        CurrentsArraySunPowerH_hyb[i,1] = SolarPowerWith_DC_DC_Conversion(voltages[i],1.0,1.0,SolarSectionsSunPowerH_hyb,NREL=True)[2]
#
#        for k in range(0,SectionsPerWing*2):
#            SolarSectionsAltaOld[k].G = 500      # W/m^2
#            SolarSectionsAlta[k].G = 500      # W/m^2
#            SolarSectionsSunPowerG[k].G = 500  # W/m^2
#            SolarSectionsSunPowerH[k].G = 500  # W/m^2
#            SolarSectionsSunPowerI[k].G = 500  # W/m^2
#            SolarSectionsSunPowerJ[k].G = 500  # W/m^2
#            SolarSectionsAlta_hyb[k].G = 500      # W/m^2
#            SolarSectionsSunPowerH_hyb[k].G = 500  # W/m^2
#        CurrentsArrayAltaOld[i,2] = SolarPowerWith_DC_DC_Conversion(voltages[i],1.0,1.0,SolarSectionsAltaOld)[2]
#        CurrentsArrayAlta[i,2] = SolarPowerWith_DC_DC_Conversion(voltages[i],1.0,1.0,SolarSectionsAlta)[2]
#        CurrentsArraySunPowerG[i,2] = SolarPowerWith_DC_DC_Conversion(voltages[i],1.0,1.0,SolarSectionsSunPowerG)[2]
#        CurrentsArraySunPowerH[i,2] = SolarPowerWith_DC_DC_Conversion(voltages[i],1.0,1.0,SolarSectionsSunPowerH)[2]
#        CurrentsArraySunPowerI[i,2] = SolarPowerWith_DC_DC_Conversion(voltages[i],1.0,1.0,SolarSectionsSunPowerI)[2]
#        CurrentsArraySunPowerJ[i,2] = SolarPowerWith_DC_DC_Conversion(voltages[i],1.0,1.0,SolarSectionsSunPowerJ)[2]
#        CurrentsArrayAlta_hyb[i,2] = SolarPowerWith_DC_DC_Conversion(voltages[i],1.0,1.0,SolarSectionsAlta_hyb,NREL=True)[2]
#        CurrentsArraySunPowerH_hyb[i,2] = SolarPowerWith_DC_DC_Conversion(voltages[i],1.0,1.0,SolarSectionsSunPowerH_hyb,NREL=True)[2]
#
#        for k in range(0,SectionsPerWing*2):
#            SolarSectionsAltaOld[k].G = 300      # W/m^2
#            SolarSectionsAlta[k].G = 300      # W/m^2
#            SolarSectionsSunPowerG[k].G = 300  # W/m^2
#            SolarSectionsSunPowerH[k].G = 300  # W/m^2
#            SolarSectionsSunPowerI[k].G = 300  # W/m^2
#            SolarSectionsSunPowerJ[k].G = 300  # W/m^2
#            SolarSectionsAlta_hyb[k].G = 300      # W/m^2
#            SolarSectionsSunPowerH_hyb[k].G = 300  # W/m^2
#        CurrentsArrayAltaOld[i,3] = SolarPowerWith_DC_DC_Conversion(voltages[i],1.0,1.0,SolarSectionsAltaOld)[2]
#        CurrentsArrayAlta[i,3] = SolarPowerWith_DC_DC_Conversion(voltages[i],1.0,1.0,SolarSectionsAlta)[2]
#        CurrentsArraySunPowerG[i,3] = SolarPowerWith_DC_DC_Conversion(voltages[i],1.0,1.0,SolarSectionsSunPowerG)[2]
#        CurrentsArraySunPowerH[i,3] = SolarPowerWith_DC_DC_Conversion(voltages[i],1.0,1.0,SolarSectionsSunPowerH)[2]
#        CurrentsArraySunPowerI[i,3] = SolarPowerWith_DC_DC_Conversion(voltages[i],1.0,1.0,SolarSectionsSunPowerI)[2]
#        CurrentsArraySunPowerJ[i,3] = SolarPowerWith_DC_DC_Conversion(voltages[i],1.0,1.0,SolarSectionsSunPowerJ)[2]
#        CurrentsArrayAlta_hyb[i,3] = SolarPowerWith_DC_DC_Conversion(voltages[i],1.0,1.0,SolarSectionsAlta_hyb,NREL=True)[2]
#        CurrentsArraySunPowerH_hyb[i,3] = SolarPowerWith_DC_DC_Conversion(voltages[i],1.0,1.0,SolarSectionsSunPowerH_hyb,NREL=True)[2]
#
#    plt.figure()
#    plt.plot(voltages, CurrentsArrayAlta[:,0], 'k-', label=r'$1000\ W/m^2$')
#    plt.plot(voltages, CurrentsArrayAlta[:,4], 'b-', label=r'$1000\ W/m^2, 50°C$')
#    plt.plot(voltages, CurrentsArrayAlta[:,1], 'r-', label=r'$800\ W/m^2$')
#    plt.plot(voltages, CurrentsArrayAlta[:,2], 'g-', label=r'$500\ W/m^2$')
#    plt.plot(voltages, CurrentsArrayAlta[:,3], 'C4-', label=r'$300\ W/m^2$')
#    plt.plot(voltages, CurrentsArrayAltaOld[:,0], 'k--')
#    plt.plot(voltages, CurrentsArrayAltaOld[:,4], 'b--')
#    plt.plot(voltages, CurrentsArrayAltaOld[:,1], 'r--')
#    plt.plot(voltages, CurrentsArrayAltaOld[:,2], 'g--')
#    plt.plot(voltages, CurrentsArrayAltaOld[:,3], 'C4--')
#    plt.ylim(0,np.amax(np.column_stack([CurrentsArrayAlta,CurrentsArrayAltaOld]))*1.1)
#    plt.xlim(0,np.amax(voltages))
#    plt.ylabel('Current (A)')
#    plt.xlabel('Voltage (V)')
#    plt.title('Array Solar Flux Variation (Parameters: New ––––  Old - - -)')
#    plt.legend(loc='lower left',prop={'size':9.5},ncol=2)
#    plt.grid()
#
#    plt.figure()
#    plt.plot(voltages, CurrentsArrayAlta[:,0], 'k-', label=r'$1000\ W/m^2$')
#    plt.plot(voltages, CurrentsArrayAlta[:,4], 'b-', label=r'$1000\ W/m^2, 50°C$')
#    plt.plot(voltages, CurrentsArrayAlta[:,1], 'r-', label=r'$800\ W/m^2$')
#    plt.plot(voltages, CurrentsArrayAlta[:,2], 'g-', label=r'$500\ W/m^2$')
#    plt.plot(voltages, CurrentsArrayAlta[:,3], 'C4-', label=r'$300\ W/m^2$')
#    plt.plot(voltages, CurrentsArrayAlta_hyb[:,0], 'k--')
#    plt.plot(voltages, CurrentsArrayAlta_hyb[:,4], 'b--')
#    plt.plot(voltages, CurrentsArrayAlta_hyb[:,1], 'r--')
#    plt.plot(voltages, CurrentsArrayAlta_hyb[:,2], 'g--')
#    plt.plot(voltages, CurrentsArrayAlta_hyb[:,3], 'C4--')
#    plt.ylim(0,np.amax(np.column_stack([CurrentsArrayAlta,CurrentsArrayAltaOld]))*1.1)
#    plt.xlim(0,np.amax(voltages))
#    plt.ylabel('Current (A)')
#    plt.xlabel('Voltage (V)')
#    plt.title('Array Solar Flux Variation (Models: Vika ––––  Hybrid - - -)')
#    plt.legend(loc='lower left',prop={'size':9.5},ncol=2)
#    plt.grid()
#
#    plt.figure()
#    plt.plot(voltages, CurrentsArraySunPowerG[:,0], 'k-', label=r'$1000\ W/m^2$')
#    plt.plot(voltages, CurrentsArraySunPowerG[:,4], 'b-', label=r'$1000\ W/m^2, 50°C$')
#    plt.plot(voltages, CurrentsArraySunPowerG[:,1], 'r-', label=r'$800\ W/m^2$')
#    plt.plot(voltages, CurrentsArraySunPowerG[:,2], 'g-', label=r'$500\ W/m^2$')
#    plt.plot(voltages, CurrentsArraySunPowerG[:,3], 'C4-', label=r'$300\ W/m^2$')
#    plt.plot(voltages, CurrentsArraySunPowerH[:,0], 'k--')
#    plt.plot(voltages, CurrentsArraySunPowerH[:,4], 'b--')
#    plt.plot(voltages, CurrentsArraySunPowerH[:,1], 'r--')
#    plt.plot(voltages, CurrentsArraySunPowerH[:,2], 'g--')
#    plt.plot(voltages, CurrentsArraySunPowerH[:,3], 'C4--')
#    plt.plot(voltages, CurrentsArraySunPowerI[:,0], 'k:')
#    plt.plot(voltages, CurrentsArraySunPowerI[:,4], 'b:')
#    plt.plot(voltages, CurrentsArraySunPowerI[:,1], 'r:')
#    plt.plot(voltages, CurrentsArraySunPowerI[:,2], 'g:')
#    plt.plot(voltages, CurrentsArraySunPowerI[:,3], 'C4:')
#    plt.plot(voltages, CurrentsArraySunPowerJ[:,0], 'k-.')
#    plt.plot(voltages, CurrentsArraySunPowerJ[:,4], 'b-.')
#    plt.plot(voltages, CurrentsArraySunPowerJ[:,1], 'r-.')
#    plt.plot(voltages, CurrentsArraySunPowerJ[:,2], 'g-.')
#    plt.plot(voltages, CurrentsArraySunPowerJ[:,3], 'C4-.')
##    plt.plot(voltages,np.ones(len(voltages))*(-1),'k-.',label='Bin J')
##    plt.plot(voltages,np.ones(len(voltages))*(-1),'k:',label='Bin I')
##    plt.plot(voltages,np.ones(len(voltages))*(-1),'k--',label='Bin H')
##    plt.plot(voltages,np.ones(len(voltages))*(-1),'k-',label='Bin G')
#    plt.ylim(0,np.amax(np.column_stack([CurrentsArraySunPowerG,CurrentsArraySunPowerH,CurrentsArraySunPowerI,CurrentsArraySunPowerJ]))*1.1)
#    plt.xlim(0,np.amax(voltages))
#    plt.ylabel('Current (A)')
#    plt.xlabel('Voltage (V)')
#    plt.title('Array Solar Flux Variation (SunPower)')
#    plt.legend(loc='lower left',prop={'size':9.5},ncol=2)
#    plt.grid()
#
#    plt.figure()
#    plt.plot(voltages, CurrentsArraySunPowerH[:,0], 'k-', label=r'$1000\ W/m^2$')
#    plt.plot(voltages, CurrentsArraySunPowerH[:,4], 'b-', label=r'$1000\ W/m^2, 50°C$')
#    plt.plot(voltages, CurrentsArraySunPowerH[:,1], 'r-', label=r'$800\ W/m^2$')
#    plt.plot(voltages, CurrentsArraySunPowerH[:,2], 'g-', label=r'$500\ W/m^2$')
#    plt.plot(voltages, CurrentsArraySunPowerH[:,3], 'C4-', label=r'$300\ W/m^2$')
#    plt.plot(voltages, CurrentsArraySunPowerH_hyb[:,0], 'k--')
#    plt.plot(voltages, CurrentsArraySunPowerH_hyb[:,4], 'b--')
#    plt.plot(voltages, CurrentsArraySunPowerH_hyb[:,1], 'r--')
#    plt.plot(voltages, CurrentsArraySunPowerH_hyb[:,2], 'g--')
#    plt.plot(voltages, CurrentsArraySunPowerH_hyb[:,3], 'C4--')
#    plt.ylim(0,np.amax(np.column_stack([CurrentsArraySunPowerG,CurrentsArraySunPowerH,CurrentsArraySunPowerI,CurrentsArraySunPowerJ]))*1.1)
#    plt.xlim(0,np.amax(voltages))
#    plt.ylabel('Current (A)')
#    plt.xlabel('Voltage (V)')
#    plt.title('Array Solar Flux Variation (Models: Vika ––––  Hybrid - - -)')
#    plt.legend(loc='lower left',prop={'size':9.5},ncol=2)
#    plt.grid()
#
#    plt.figure()
#    plt.plot(voltages, CurrentsArraySunPowerH[:,0], 'k--', label=r'$1000\ W/m^2$')
#    plt.plot(voltages, CurrentsArraySunPowerH[:,4], 'b--', label=r'$1000\ W/m^2, 50°C$')
#    plt.plot(voltages, CurrentsArraySunPowerH[:,1], 'r--', label=r'$800\ W/m^2$')
#    plt.plot(voltages, CurrentsArraySunPowerH[:,2], 'g--', label=r'$500\ W/m^2$')
#    plt.plot(voltages, CurrentsArraySunPowerH[:,3], 'C4--', label=r'$300\ W/m^2$')
#    plt.plot(voltages, CurrentsArrayAlta[:,0], 'k-')
#    plt.plot(voltages, CurrentsArrayAlta[:,4], 'b-')
#    plt.plot(voltages, CurrentsArrayAlta[:,1], 'r-')
#    plt.plot(voltages, CurrentsArrayAlta[:,2], 'g-')
#    plt.plot(voltages, CurrentsArrayAlta[:,3], 'C4-')
#    plt.ylim(0,np.amax(np.column_stack([CurrentsArrayAlta,CurrentsArraySunPowerH]))*1.1)
#    plt.xlim(0,np.amax(voltages))
#    plt.ylabel('Current (A)')
#    plt.xlabel('Voltage (V)')
#    plt.title('Array Solar Flux Variation (Cell Type: Alta ––––  SunPower - - -)')
#    plt.legend(loc='lower left',prop={'size':8.5},ncol=2)
#    plt.grid()
#
#    ## Efficiency vs. Flux Plots - Whole Array
#    T = np.array([25,0,-25,-50,-75]) + 273.15  # (K)
#    nlines = len(T)
#    flux = np.linspace(0,1200,150) # (W/m^2)
#    AltaOld_ArrayEfficiencies = np.zeros((len(flux),nlines))
#    Alta_ArrayEfficiencies = np.zeros((len(flux),nlines))
#    SunPower_ArrayEfficienciesG = np.zeros((len(flux),nlines))
#    SunPower_ArrayEfficienciesH = np.zeros((len(flux),nlines))
#    SunPower_ArrayEfficienciesI = np.zeros((len(flux),nlines))
#    SunPower_ArrayEfficienciesJ = np.zeros((len(flux),nlines))
#    Alta_ArrayEfficiencies_hyb = np.zeros((len(flux),nlines))
#    SunPower_ArrayEfficienciesH_hyb = np.zeros((len(flux),nlines))
#
#    for i in range(0,len(flux)):
#        for k in range(0,SectionsPerWing*2):
#            SolarSectionsAltaOld[k].G = flux[i]
#            SolarSectionsAlta[k].G = flux[i]
#            SolarSectionsSunPowerG[k].G = flux[i]
#            SolarSectionsSunPowerH[k].G = flux[i]
#            SolarSectionsSunPowerI[k].G = flux[i]
#            SolarSectionsSunPowerJ[k].G = flux[i]
#            SolarSectionsAlta_hyb[k].G = flux[i]
#            SolarSectionsSunPowerH_hyb[k].G = flux[i]
#        for j in range(0,nlines):
#            for k in range(0,SectionsPerWing*2):
#                SolarSectionsAltaOld[k].T = T[j]
#                SolarSectionsAlta[k].T = T[j]
#                SolarSectionsSunPowerG[k].T = T[j]
#                SolarSectionsSunPowerH[k].T = T[j]
#                SolarSectionsSunPowerI[k].T = T[j]
#                SolarSectionsSunPowerJ[k].T = T[j]
#                SolarSectionsAlta_hyb[k].T = T[j]
#                SolarSectionsSunPowerH_hyb[k].T = T[j]
#            AltaOld_ArrayEfficiencies[i][j] = SolarPowerMPP(SolarSectionsAltaOld)[8]
#            Alta_ArrayEfficiencies[i][j] = SolarPowerMPP(SolarSectionsAlta)[8]
#            SunPower_ArrayEfficienciesG[i][j] = SolarPowerMPP(SolarSectionsSunPowerG)[8]
#            SunPower_ArrayEfficienciesH[i][j] = SolarPowerMPP(SolarSectionsSunPowerH)[8]
#            SunPower_ArrayEfficienciesI[i][j] = SolarPowerMPP(SolarSectionsSunPowerI)[8]
#            SunPower_ArrayEfficienciesJ[i][j] = SolarPowerMPP(SolarSectionsSunPowerJ)[8]
#            Alta_ArrayEfficiencies_hyb[i][j] = SolarPowerMPP(SolarSectionsAlta_hyb,NREL=True)[8]
#            SunPower_ArrayEfficienciesH_hyb[i][j] = SolarPowerMPP(SolarSectionsSunPowerH_hyb,NREL=True)[8]
#    for i in range(0,len(flux)-1):
#        if flux[i+1] > 1000 and flux[i] < 1000:
#            Flux1000_25_AltaOld = AltaOld_ArrayEfficiencies[i,0]
#            Flux1000_25_Alta = Alta_ArrayEfficiencies[i,0]
#            Flux1000_25_SPG = SunPower_ArrayEfficienciesG[i,0]
#            Flux1000_25_SPH = SunPower_ArrayEfficienciesH[i,0]
#            Flux1000_25_SPI = SunPower_ArrayEfficienciesI[i,0]
#            Flux1000_25_SPJ = SunPower_ArrayEfficienciesJ[i,0]
#            Flux1000_25_Alta_hyb = Alta_ArrayEfficiencies_hyb[i,0]
#            Flux1000_25_SPH_hyb = SunPower_ArrayEfficienciesH_hyb[i,0]
#            break
#
#    plt.figure()
#    plt.title("Alta Devices Array Efficiencies vs. Flux (Parameters: New ––––  Old - - -)")
#    plt.plot(flux,Alta_ArrayEfficiencies[:,0],'g-',label=str(int(T[0]-273.15))+" °C")
#    plt.plot(flux,Alta_ArrayEfficiencies[:,1],'b-',label=str(int(T[1]-273.15))+" °C")
#    plt.plot(flux,Alta_ArrayEfficiencies[:,2],'r-',label=str(int(T[2]-273.15))+" °C")
#    plt.plot(flux,Alta_ArrayEfficiencies[:,3],'c-',label=str(int(T[3]-273.15))+" °C")
#    plt.plot(flux,Alta_ArrayEfficiencies[:,4],'m-',label=str(int(T[4]-273.15))+" °C")
#    plt.plot(flux,AltaOld_ArrayEfficiencies[:,0],'g--')
#    plt.plot(flux,AltaOld_ArrayEfficiencies[:,1],'b--')
#    plt.plot(flux,AltaOld_ArrayEfficiencies[:,2],'r--')
#    plt.plot(flux,AltaOld_ArrayEfficiencies[:,3],'c--')
#    plt.plot(flux,AltaOld_ArrayEfficiencies[:,4],'m--')
#    plt.plot(1000,Flux1000_25_AltaOld,'go',alpha=0.9)
#    plt.plot(1000,Flux1000_25_Alta,'go')
#    plt.text(200,0.16, 'Old: %.2f' % (Flux1000_25_AltaOld*100) +' %',color='g',ha='left')
#    plt.text(200,0.153,'New: %.2f' % (Flux1000_25_Alta*100) +' %',color='g',ha='left')
#    plt.xlim([0,flux[-1]])
#    plt.ylim([0.15,0.30])
#    plt.ylabel("Efficiency")
#    plt.xlabel("Flux (W/m^2)")
#    plt.legend(loc=4, ncol = 2)
#
#    plt.figure()
#    plt.title("Alta Devices Array Efficiencies vs. Flux (Models: Vika ––––  Hybrid - - -)")
#    plt.plot(flux,Alta_ArrayEfficiencies[:,0],'g-',label=str(int(T[0]-273.15))+" °C")
#    plt.plot(flux,Alta_ArrayEfficiencies[:,1],'b-',label=str(int(T[1]-273.15))+" °C")
#    plt.plot(flux,Alta_ArrayEfficiencies[:,2],'r-',label=str(int(T[2]-273.15))+" °C")
#    plt.plot(flux,Alta_ArrayEfficiencies[:,3],'c-',label=str(int(T[3]-273.15))+" °C")
#    plt.plot(flux,Alta_ArrayEfficiencies[:,4],'m-',label=str(int(T[4]-273.15))+" °C")
#    plt.plot(flux,Alta_ArrayEfficiencies_hyb[:,0],'g--')
#    plt.plot(flux,Alta_ArrayEfficiencies_hyb[:,1],'b--')
#    plt.plot(flux,Alta_ArrayEfficiencies_hyb[:,2],'r--')
#    plt.plot(flux,Alta_ArrayEfficiencies_hyb[:,3],'c--')
#    plt.plot(flux,Alta_ArrayEfficiencies_hyb[:,4],'m--')
#    plt.plot(1000,Flux1000_25_Alta_hyb,'go',alpha=0.9)
#    plt.plot(1000,Flux1000_25_Alta,'go')
#    plt.text(200,0.16, 'Hybrid: %.2f' % (Flux1000_25_Alta_hyb*100) +' %',color='g',ha='left')
#    plt.text(200,0.153,'   Vika: %.2f' % (Flux1000_25_Alta*100) +' %',color='g',ha='left')
#    plt.xlim([0,flux[-1]])
#    plt.ylim([0.15,0.30])
#    plt.ylabel("Efficiency")
#    plt.xlabel("Flux (W/m^2)")
#    plt.legend(loc=4, ncol = 2)
#
#    plt.figure()
#    plt.title("Array Efficiencies vs. Flux (Cell Type: Alta ––––  SunPower - - -)")
#    plt.plot(flux,Alta_ArrayEfficiencies[:,0],'g-',label=str(int(T[0]-273.15))+" °C")
#    plt.plot(flux,Alta_ArrayEfficiencies[:,1],'b-',label=str(int(T[1]-273.15))+" °C")
#    plt.plot(flux,Alta_ArrayEfficiencies[:,2],'r-',label=str(int(T[2]-273.15))+" °C")
#    plt.plot(flux,Alta_ArrayEfficiencies[:,3],'c-',label=str(int(T[3]-273.15))+" °C")
#    plt.plot(flux,Alta_ArrayEfficiencies[:,4],'m-',label=str(int(T[4]-273.15))+" °C")
#    plt.plot(flux,SunPower_ArrayEfficienciesH[:,0],'g--')
#    plt.plot(flux,SunPower_ArrayEfficienciesH[:,1],'b--')
#    plt.plot(flux,SunPower_ArrayEfficienciesH[:,2],'r--')
#    plt.plot(flux,SunPower_ArrayEfficienciesH[:,3],'c--')
#    plt.plot(flux,SunPower_ArrayEfficienciesH[:,4],'m--')
#    plt.plot(1000,Flux1000_25_Alta,'go',alpha=0.9)
#    plt.plot(1000,Flux1000_25_SPH,'go')
#    plt.text(200,0.16, 'Alta: %.2f' % (Flux1000_25_Alta*100) +' %',color='g',ha='left')
#    plt.text(200,0.153,'SunPower: %.2f' % (Flux1000_25_SPH*100) +' %',color='g',ha='left')
#    plt.xlim([0,flux[-1]])
#    plt.ylim([0.15,0.30])
#    plt.ylabel("Efficiency")
#    plt.xlabel("Flux (W/m^2)")
#    plt.legend(loc=4, ncol = 2)
#
#    plt.figure()
#    plt.title("SunPower Array Efficiencies vs. Flux")
#    plt.plot(flux,SunPower_ArrayEfficienciesG[:,4],'m-',label=str(int(T[4]-273.15))+" °C")
#    plt.plot(flux,SunPower_ArrayEfficienciesG[:,3],'c-',label=str(int(T[3]-273.15))+" °C")
#    plt.plot(flux,SunPower_ArrayEfficienciesG[:,2],'r-',label=str(int(T[2]-273.15))+" °C")
#    plt.plot(flux,SunPower_ArrayEfficienciesG[:,1],'b-',label=str(int(T[1]-273.15))+" °C")
#    plt.plot(flux,SunPower_ArrayEfficienciesG[:,0],'g-',label=str(int(T[0]-273.15))+" °C")
#    plt.plot(flux,SunPower_ArrayEfficienciesH[:,0],'g--')
#    plt.plot(flux,SunPower_ArrayEfficienciesH[:,1],'b--')
#    plt.plot(flux,SunPower_ArrayEfficienciesH[:,2],'r--')
#    plt.plot(flux,SunPower_ArrayEfficienciesH[:,3],'c--')
#    plt.plot(flux,SunPower_ArrayEfficienciesH[:,4],'m--')
#    plt.plot(flux,SunPower_ArrayEfficienciesI[:,0],'g:')
#    plt.plot(flux,SunPower_ArrayEfficienciesI[:,1],'b:')
#    plt.plot(flux,SunPower_ArrayEfficienciesI[:,2],'r:')
#    plt.plot(flux,SunPower_ArrayEfficienciesI[:,3],'c:')
#    plt.plot(flux,SunPower_ArrayEfficienciesI[:,4],'m:')
#    plt.plot(flux,SunPower_ArrayEfficienciesJ[:,0],'g-.')
#    plt.plot(flux,SunPower_ArrayEfficienciesJ[:,1],'b-.')
#    plt.plot(flux,SunPower_ArrayEfficienciesJ[:,2],'r-.')
#    plt.plot(flux,SunPower_ArrayEfficienciesJ[:,3],'c-.')
#    plt.plot(flux,SunPower_ArrayEfficienciesJ[:,4],'m-.')
#    plt.plot(flux,np.ones(len(flux)),'k-.',label='Bin J')
#    plt.plot(flux,np.ones(len(flux)),'k:',label='Bin I')
#    plt.plot(flux,np.ones(len(flux)),'k--',label='Bin H')
#    plt.plot(flux,np.ones(len(flux)),'k-',label='Bin G')
#    plt.plot(1000,Flux1000_25_SPG,'go')
#    plt.plot(1000,Flux1000_25_SPH,'go')
#    plt.plot(1000,Flux1000_25_SPI,'go')
#    plt.plot(1000,Flux1000_25_SPJ,'go')
#    plt.text(200,0.174,'J: %.2f' % (Flux1000_25_SPJ*100) +' %',color='g',ha='left')
#    plt.text(200,0.167,'I: %.2f' % (Flux1000_25_SPI*100) +' %',color='g',ha='left')
#    plt.text(200,0.16,'H: %.2f' % (Flux1000_25_SPH*100) +' %',color='g',ha='left')
#    plt.text(200,0.153,'G: %.2f' % (Flux1000_25_SPG*100) +' %',color='g',ha='left')
#    plt.xlim([0,flux[-1]])
#    plt.ylim([0.15,0.30])
#    plt.ylabel("Efficiency")
#    plt.xlabel("Flux (W/m^2)")
#    plt.legend(loc=4, ncol = 2)
#
#    plt.figure()
#    plt.title("SunPower Array Efficiencies vs. Flux (Models: Vika ––––  Hybrid - - -)")
#    plt.plot(flux,SunPower_ArrayEfficienciesH[:,4],'m-',label=str(int(T[4]-273.15))+" °C")
#    plt.plot(flux,SunPower_ArrayEfficienciesH[:,3],'c-',label=str(int(T[3]-273.15))+" °C")
#    plt.plot(flux,SunPower_ArrayEfficienciesH[:,2],'r-',label=str(int(T[2]-273.15))+" °C")
#    plt.plot(flux,SunPower_ArrayEfficienciesH[:,1],'b-',label=str(int(T[1]-273.15))+" °C")
#    plt.plot(flux,SunPower_ArrayEfficienciesH[:,0],'g-',label=str(int(T[0]-273.15))+" °C")
#    plt.plot(flux,SunPower_ArrayEfficienciesH_hyb[:,0],'g--')
#    plt.plot(flux,SunPower_ArrayEfficienciesH_hyb[:,1],'b--')
#    plt.plot(flux,SunPower_ArrayEfficienciesH_hyb[:,2],'r--')
#    plt.plot(flux,SunPower_ArrayEfficienciesH_hyb[:,3],'c--')
#    plt.plot(flux,SunPower_ArrayEfficienciesH_hyb[:,4],'m--')
#    plt.plot(1000,Flux1000_25_SPH,'go')
#    plt.plot(1000,Flux1000_25_SPH_hyb,'go')
#    plt.text(200,0.16, 'Hybrid: %.2f' % (Flux1000_25_SPH_hyb*100) +' %',color='g',ha='left')
#    plt.text(200,0.153,'   Vika: %.2f' % (Flux1000_25_SPH*100) +' %',color='g',ha='left')
#    plt.xlim([0,flux[-1]])
#    plt.ylim([0.15,0.30])
#    plt.ylabel("Efficiency")
#    plt.xlabel("Flux (W/m^2)")
#    plt.legend(loc=4, ncol = 2)

"""
############################################ OLD - Faster Solar Power Functions ###################################################################################################
def IandV(X,G,N_p,R_p,R_s,V_t,a,I_ph,I_0):
    I_pv = X[0:-1]    # (A) [array with all initial guess values for current of cells in series]
    V_pv = X[-1]      # (V) [guess value for optimal voltage the array is operating at - "voltage output of the series coupling" (Vika)]
    # G - (W/m^2,[number of solar sections]) [array of solar fluxes associated with each solar panel section]
    # N_p - (unitless,[number of solar sections]) [Number of strings of solar cells in parallel associated with each solar panel section]
    # R_p - (Ohms) [Equivalent parallel resistance of a single cell - Alta Devices]
    # R_s - (Ohms) [Equivalent series resistance of a single cell - Alta Devices]
    # V_t - (V,[number of solar sections]) [Thermal voltage of solar cells in series (Vika page 22)]
    # a - [Ideality Factor]
    # I_ph - (A,[number of solar sections]) [Photo Current of solar cells in series - accounts for the difference in temperature and flux from 25°C and 1000 W/m^2]
    # I_0 - (A,[number of solar sections])

    n = len(X)
    zeros = np.zeros(n)
    for i in range(0,n-1):
        if G[i] <= 0:
            if I_pv[i] == 0:
                zeros[i] = 0
            else:
                zeros[i] = 1000
        else:
            zeros[i] = I_ph[i] - I_0[i]*(np.exp((V_pv+R_s*I_pv[i])/(V_t[i]*a))-1) - (V_pv + R_s*I_pv[i])/R_p - I_pv[i]

#    for i in range(0,n-1):
#        if I_pv[i] != 0 and G[i] > 0:
#            zeros[i] = I_ph[i] - I_0[i]*(np.exp((V_pv+R_s*I_pv[i])/(V_t[i]*a))-1) - (V_pv + R_s*I_pv[i])/R_p - I_pv[i]

#    zeros[:-1] = I_ph - I_0*(np.exp((V_pv+R_s*I_pv)/(V_t*a))-1) - (V_pv + R_s*I_pv)/R_p - I_pv

#    I_pv[I_pv < 0] = 0 # IS THIS CHECK NEEDED? - Blocking diodes prevent negative currents
    I = I_pv*N_p
    tracker = 0
    for i in range(0,n-1):
        if G[i] > 0 and I_pv[i] > 0:
            zeros[-1] += I[i]-N_p[i]*V_pv/(-I_0[i]*R_s/(V_t[i]*a)*np.exp((V_pv+R_s*I_pv[i])/(V_t[i]*a))-R_s/R_p-1)*(-I_0[i]/(V_t[i]*a)*np.exp((V_pv+R_s*I_pv[i])/(V_t[i]*a))-1/R_p)
            tracker += 1
    if tracker == 0:
        zeros[-1] = 1000

    #zeros[-1] = sum(I-N_p*V_pv/(-I_0*R_s/(V_t*a)*np.exp((V_pv+R_s*I_pv)/(V_t*a))-R_s/R_p-1)*(-I_0/(V_t*a)*np.exp((V_pv+R_s*I_pv)/(V_t*a))-1/R_p)) # dP/dV by analytic sensitivity equations
    #zeros[-1] = sum(1 + (R_s - N_p*V_pv/I) * (I_0/(V_t*a) * np.exp((V_pv + R_s*I_pv)/(V_t*a)) + 1/R_p)) # dP/dV (simplified) by analytic sensitivity equations - DOES NOT WORK RIGHT NOW
        # 0 = 1 + (R_s - N_p*V_pv/I)*(I_0/(V_t*a)*exp((V_pv + R_s*I_pv)/(V_t*a)) + 1/R_p) ! dP/dV (simplified) by analytic sensitivity equations

    return zeros

def SolarPowerMPP_Direct(parameters_group,num_divisions,T,G,SolarPanelAreas,N_s,N_p,R_p,R_s,V_oc,I_sc,T_coef_V,T_coef_I,a,I_ph_ref,q,k_B,T_n,G_n,
                         K_V,K_I,I_pv_guess,high_efficiency=1.0,efficiency=1.0,UseAltaTempData=False):
    # parameters_group - [array containing several Parameter objects]
    # num_divisions - [number of solar sections]
    # T - (K,[number of solar sections]) [array of solar panel temperatures associated with each solar panel section]
    # G - (W/m^2,[number of solar sections]) [array of solar fluxes associated with each solar panel section]
    # high_efficiency - [efficiency of the converter when input and output voltages are practically the same]
    # SolarPanelAreas - (m^2,[number of solar sections]) [array of the areas of each solar panel section]
    # N_s - (unitless) [Number of cells in series - This is not an array because the number of cells in series must be the same for all strings if the open circuit voltage is the same for all strings]
    # N_p - (unitless,[number of solar sections]) [Number of strings of solar cells in parallel associated with each solar panel section]
    # R_p - (Ohms) [Equivalent parallel resistance of a single cell - Alta Devices]
    # R_s - (Ohms) [Equivalent series resistance of a single cell - Alta Devices]
    # V_oc - (V) [Series Open-Circuit Voltage]
    # I_sc - (A) [Short Circuit Current of Single Cell]
    # T_coef_V - (%/K) [Voltage Temperature Coefficient - percent change per °C from 25°C]
    # T_coef_I - (%/K) [Current Temperature Coefficient - percent change per °C from 25°C]
    # a - [Ideality Factor]
    # I_ph_ref - (A) [Photo Current of a single cell at 25°C and 1000 W/m^2 - Alta Devices]
    # q - (C) [Charge of an electron]
    # k_B - (J/K) [Boltzmann constant]
    # T_n - (K)
    # G_n - (W/m^2)
    # K_V - (V/K)
    # K_I - (A/K)
    # I_pv_guess - (A) [guess value for the current of a single string of solar cells in series]
    # efficiency - [efficiency of the converter in other conditions]
    total_flux = sum(G)                               # (W/m^2) [Flux hitting the entire array]
    voltage_out = parameters_group[0].SolarVoltageOpenCircuit # (V) [voltage exiting the converter]

    if total_flux <= 0:
        Vopt = 0
        solar_efficiency = 0
        solar_efficiency_after_DC_DC = 0
        [power_out,power_in,current_in,solar_section_efficiencies] = [0,0,0,np.zeros(num_divisions)]
    else:
        Delta_T = T-T_n
        V_t = N_s*k_B*T/q # (V,[number of solar sections]) [Thermal voltage of solar cells in series (Vika page 22)]
        if UseAltaTempData == True:
            Delta_T = 0
            V_oc = (-0.0015*(T-273.15)+1.1547)*N_s # V (Voltage from data fit given to us by Alta Devices)
            I_sc = (0.0925*(T-273.15)+328.03)/1000 # A (Current from data fit given to us by Alta Devices)
            V_t = N_s*k_B*T_n/q # Changed T to T_n (V) [Thermal voltage of solar cells in series (Vika page 22)]
            K_V = T_coef_V/100*V_oc # (V/K)
            K_I = T_coef_I/100*I_sc # (A/K)
        I_ph = (I_ph_ref + K_I*Delta_T)*(G/G_n) # (A,[number of solar sections]) [Photo Current of solar cells in series - accounts for the difference in temperature and flux from 25°C and 1000 W/m^2]
        I_0 = (I_sc + K_I*Delta_T)/(np.exp((V_oc+K_V*Delta_T)/(a*V_t))-1) # (A,[number of solar sections])

        total_flux_power = sum(G * SolarPanelAreas)   # (W) [Flux power hitting the entire array]

        arg = (G,N_p,R_p,R_s,V_t,a,I_ph,I_0)
        guesses = np.ones(num_divisions+1)*I_pv_guess     # (A) [array with all initial guess values for current of cells in series]
        for i in range(0,num_divisions):
            if G[i] <= 0:
                guesses[i] = 0

        guesses[-1] = voltage_out                         # (V) [guess value for optimal voltage the array is operating at - "voltage output of the series coupling" (Vika)]
        Results_w_message = fsolve(IandV,guesses,args=arg,full_output=True)
        Results = Results_w_message[0]

        Vopt = Results[-1] # (V) [Optimal operating voltage of the solar array]
        #voltage_in = Vopt                                        # (V) [voltage entering the converter]
        solar_section_efficiencies = np.zeros(num_divisions)      # (array, unitless) [efficiencies of each solar section]
        currents_single = Results[:-1] * N_p                      # (array,A) [currents from each solar section]
        currents_single[currents_single < 0] = 0
        current_in = sum(currents_single)                      # (A) [Current entering the converter from entire solar array]
        for i in range(0,num_divisions):
            if G[i] > 0:
                solar_section_efficiencies[i] = Vopt*currents_single[i]/(G[i]*SolarPanelAreas[i])

        power_in = Vopt*current_in                # (W) [power entering the converter]
        # If voltages are practically the same, efficiency is very high
        if Vopt-voltage_out > - 1.0 and Vopt-voltage_out < 1.0:
            power_out = power_in*high_efficiency  # (W) [power exiting the converter]
        else:
            power_out = power_in*efficiency       # (W) [power exiting the converter]

        # This won't work - need to update for new functions
        if efficiency < 1.0: # Fixes error where the brent function doesn't always catch the high DC_DC conversion efficiency - NOT PERFECT
            [other1,other2,other3,other4] = SolarPowerWith_DC_DC_Conversion(voltage_out,high_efficiency,efficiency,parameters_group,UseAltaTempData)
            if other1 > power_out:
                [power_out,power_in,current_in,solar_section_efficiencies] = [other1,other2,other3,other4]
                Vopt = voltage_out

        power_out = power_out * 0.983  # (W) [98.3% efficiency - according to peak power trackers from Facebook]
        solar_efficiency = power_in/total_flux_power
        solar_efficiency_after_DC_DC = power_out/total_flux_power


        [old_power_in,old_current_in,old_vopt] = SolarPowerMPP(parameters_group,high_efficiency=1.0,efficiency=1.0,UseAltaTempData=False)[4:7]
        if Results_w_message[2] != 1:
            print (Results_w_message[2:])
            print ('\t',I_ph)
            print ('\t',T)
            print ('\t',G)
        for j in range(0,num_divisions):
            if parameters_group[j].T != T[j]:
                print ("\t\tTemperature Not Equal:",parameters_group[j].T,T[j])
            if parameters_group[j].G != G[j]:
                print ("\t\tFlux Not Equal:",parameters_group[j].G,G[j])
        if abs(power_in-old_power_in)/old_power_in > 0.6/100:
            print ("Power Not Equal!")
            print ('\t',power_in,old_power_in)
            print ('\tPercent Error:',abs(power_in-old_power_in)*100/old_power_in,'%')
        if abs(Vopt - old_vopt)/old_vopt > 0.6/100:
            print ("Voltage Not Equal!")
            print ('\t',Vopt,old_vopt)
            print ('\tPercent Error:',abs(Vopt-old_vopt)*100/old_vopt,'%')
            # Test Plots
            voltages = np.linspace(0,150,150) # V
            powers = np.empty(len(voltages))  # W

            for j in range(0,len(voltages)):
                powers[j] = SolarPowerWith_DC_DC_Conversion(voltages[j],1.0,1.0,parameters_group,UseAltaTempData=False)[0] # W
            plt.figure()
            plt.plot(voltages,powers)
            plt.plot(Vopt,power_in,'o')
            plt.plot(old_vopt,old_power_in,'s')
            plt.xlabel("Voltage (V)")
            plt.ylabel("Power (W)")
            plt.title("Power vs. Voltage")
        if abs(current_in-old_current_in)/old_current_in > 0.6/100:
            print ("Current Not Equal!")
            print ('\t',current_in,old_current_in)
            print ('\tPercent Error:',abs(current_in-old_current_in)*100/old_current_in,'%')

            Results_old = np.empty(num_divisions+1)
            Results_old[-1] = old_vopt
            for o in range(0,num_divisions):
                parameters_group[o].V_pv = Results_old[-1]
                Results_old[o] = I_Single_String(parameters_group[o],UseAltaTempData,ReturnNegative=True)
                print (Results[o],Results_old[o])
            zeros_new = IandV(Results,G,N_p,R_p,R_s,V_t,a,I_ph,I_0)
            zeros_old = IandV(Results_old,G,N_p,R_p,R_s,V_t,a,I_ph,I_0)
            for i in range(0,num_divisions+1):
                print ('\t\t',zeros_new[i],zeros_old[i],'\n')
                if i == num_divisions and (zeros_old[i] > 1e-4 or zeros_old[i] < -1e-4):
                    print ("\n************************************Voltage didn't solve IandV successfully!************************************\n")
                if i < num_divisions and (zeros_old[i] > 1e-4 or zeros_old[i] < -1e-4):
                    I_Single_String(parameters_group[i],PrintF=True)

    current_out = power_out/voltage_out       # (A) [current exiting the converter]
    power_lost = power_in-power_out         # (W) [power lost to the converter and MPP tracker]
    return power_out, current_out, voltage_out, solar_efficiency_after_DC_DC, power_in, current_in, Vopt, power_lost, solar_efficiency, solar_section_efficiencies
    # power_out - (W) [Power exiting the converter]
    # current_out - (A) [Current exiting the converter]
    # voltage_out - (V) [Voltage exiting the converter - should be the system voltage]
    # solar_efficiency_after_DC_DC - (unitless) [Efficiency of solar panels after DC-DC conversion]
    # power_in - (W) [Power entering the converter from the solar array]
    # current_in - (A) [Current entering the converter from entire solar array]
    # Vopt - (V) [Optimal operating voltage of the solar array]
    # power_lost - (W) [Power lost to the converter and MPP tracker]
    # solar_efficiency - (unitless) [Efficiency of solar panels]
    # solar_section_efficiencies - (array, unitless) [efficiencies of each solar section]


def Simulation_Direct(time,temperatures,fluxes,parameters_group,I_pv_guess=0.05,high_efficiency=1.0,efficiency=1.0,UseAltaTempData=False):
    # time - (hours) [linear array of times]
    # fluxes - (W/m^2,[number of solar sections, length of time array]) [array of solar fluxes associated with the time array and each solar panel section]
    # temperatures - (K,[number of solar sections, length of time array]) [array of solar panel temperatures associated with the time array and each solar panel section]
    # parameters_group - [array containing several Parameter objects]
    # I_pv_guess - (A) [guess value for the current of a single string of solar cells]
    # high_efficiency - [efficiency of the converter when input and output voltages are practically the same]
    # efficiency - [efficiency of the converter in other conditions]
    # UseAltaTempData - (bool) [removes temperature dependence from model and uses fit from Alta Devices data instead]
    num_divisions = len(parameters_group)   # number of solar sections

    # Time Independent Arrays
    SolarPanelAreas = np.empty(num_divisions) # (m^2) [Solar panel section areas]
    N_p = np.empty(num_divisions)             # [Number of strings in parallel]
    for i in range(0,num_divisions):
        SolarPanelAreas[i] = parameters_group[i].SolarCellArea
        N_p[i] = parameters_group[i].N_p

    # Constants (All Independent of Time)
    N_s = parameters_group[0].N_s                      # [Number of cells in series - This is not an array because the number of cells in series must be the same for all strings if the open circuit voltage is the same for all strings]
    R_p = parameters_group[0].R_p                      # (Ohms) [Equivalent parallel resistance of a single cell - Alta Devices]
    R_s = parameters_group[0].R_s                      # (Ohms) [Equivalent series resistance of a single cell - Alta Devices]
    V_oc = parameters_group[0].SolarVoltageOpenCircuit # (V) [Series Open-Circuit Voltage]
    I_sc = parameters_group[0].I_sc                    # (A) [Short Circuit Current of Single Cell]
    T_coef_V = parameters_group[0].T_coef_V            # (%/K) [Voltage Temperature Coefficient - percent change per °C from 25°C]
    T_coef_I = parameters_group[0].T_coef_I            # (%/K) [Current Temperature Coefficient - percent change per °C from 25°C]
    a = parameters_group[0].a                          # [Ideality Factor]
    I_ph_ref = parameters_group[0].I_ph_ref            # (A) [Photo Current of a single cell at 25°C and 1000 W/m^2 - Alta Devices]
    q = 1.60217646e-19   # (C) [Charge of an electron]
    k_B = 1.38064852e-23 # (J/K) [Boltzmann constant]
    T_n = 25 + 273.15    # (K)
    G_n = 1000           # (W/m^2)

    R_p = N_s*R_p # (Ohms) [Equivalent parallel resistance of solar cells in series - Alta Devices]
    R_s = N_s*R_s # (Ohms) [Equivalent series resistance of solar cells in series - Alta Devices]
    K_V = T_coef_V/100*V_oc # (V/K)
    K_I = T_coef_I/100*I_sc # (A/K)

    # Initialize Arrays
    power_out = np.empty(len(time))
    current_out = np.empty(len(time))
    voltage_out = np.empty(len(time))
    solar_efficiency_after_DC_DC = np.empty(len(time))
    power_in = np.empty(len(time))
    current_in = np.empty(len(time))
    Vopt = np.empty(len(time))
    power_lost = np.empty(len(time))
    solar_efficiency = np.empty(len(time))
    solar_section_efficiencies = np.empty((num_divisions,len(time)))

    for i in range(0,len(time)):
        for j in range(0,num_divisions): # These can be commented out when not troubleshooting
            parameters_group[j].T = temperatures[j,i] # These can be commented out when not troubleshooting
            parameters_group[j].G = fluxes[j,i] # These can be commented out when not troubleshooting
        [power_out[i], current_out[i], voltage_out[i], solar_efficiency_after_DC_DC[i],
         power_in[i], current_in[i], Vopt[i], power_lost[i], solar_efficiency[i],
         solar_section_efficiencies[:,i]] = SolarPowerMPP_Direct(parameters_group,num_divisions,temperatures[:,i],fluxes[:,i],SolarPanelAreas,N_s,N_p,R_p,R_s,V_oc,I_sc,
                                         T_coef_V,T_coef_I,a,I_ph_ref,q,k_B,T_n,G_n,K_V,K_I,I_pv_guess,high_efficiency,efficiency,UseAltaTempData)
    return power_out, current_out, voltage_out, solar_efficiency_after_DC_DC, power_in, current_in, Vopt, power_lost, solar_efficiency, solar_section_efficiencies
    # power_out - (W) [Power exiting the converter]
    # current_out - (A) [Current exiting the converter]
    # voltage_out - (V) [Voltage exiting the converter - should be the system voltage]
    # solar_efficiency_after_DC_DC - (unitless) [Efficiency of solar panels after DC-DC conversion]
    # power_in - (W) [Power entering the converter from the solar array]
    # current_in - (A) [Current entering the converter from entire solar array]
    # Vopt - (V) [Optimal operating voltage of the solar array]
    # power_lost - (W) [Power lost to the converter and MPP tracker]
    # solar_efficiency - (unitless) [Efficiency of solar panels]
    # solar_section_efficiencies - (array, unitless) [efficiencies of each solar section]

# Analyze faster (direct) functions
if __name__ == "__main__":
    SolarPanelSections = []
    for i in range(0,42):
        SolarPanelSections.append(Parameter(T=-56.45+273.15,G=1000-i*25,Area_Desired=0.72254))
        #print(I_Single_String(SolarPanelSections[i]),SolarPanelSections[i].T,SolarPanelSections[i].G)
        #print (SolarPanelSections[i].N_s,SolarPanelSections[i].N_p,SolarPanelSections[i].R_s,SolarPanelSections[i].R_p)

    time = np.linspace(0,24,250) # hr
    temperature = np.linspace(-65,-35,len(time))+273.15 # K
    flux = np.linspace(500,1000,len(time)) # W/m^2
    temperatures = np.empty((len(SolarPanelSections),len(time)))
    fluxes = np.empty((len(SolarPanelSections),len(time)))
    for j in range(0,len(SolarPanelSections)):
        temperatures[j,:] = temperature+j*5
        fluxes[j,:] = flux-600/len(SolarPanelSections)*(j+1)
    results_old = np.empty((len(time),9))
    efficiencies_old = np.empty((len(SolarPanelSections),len(time)))
    time_direct = 0
    time_old = 0

    start = t.time()
    new_results = Simulation_Direct(time,temperatures,fluxes,SolarPanelSections,high_efficiency=1.0,efficiency=1.0,UseAltaTempData=False)
    results_direct = np.column_stack([new_results[:-1]]).T
    efficiencies_direct = new_results[-1]
    end = t.time()
    time_direct += end-start
#    for i in range(0,len(time)):
#        for j in range(0,len(SolarPanelSections)):
#            SolarPanelSections[j].T = temperatures[j,i]
#            SolarPanelSections[j].G = fluxes[j,i]
#        start = t.time()
#        old_results = SolarPowerMPP(SolarPanelSections,UseAltaTempData=False)
#        results_old[i,:] = old_results[:-1]
#        efficiencies_old[:,i] = old_results[-1]
#        end = t.time()
#        time_old += end-start
#        for j in range(0,9):
#            if abs(results_direct[i,j] - results_old[i,j]) > 2.7e-3:
#                print ('NOT EQUAL',i,j)
#                print ('\t',results_direct[i,j],results_old[i,j])
#                print ('\tPercent Error',("%.10f" % ((results_direct[i,j]-results_old[i,j])*100/results_old[i,j])),'%')
#            if j != 0 and j != 4 and abs(results_direct[i,j] - results_old[i,j]) > 4.8e-5:
#                print ('NOT EQUAL',i,j)
#                print ('\t',results_direct[i,j],results_old[i,j])
#                print ('\tPercent Error:',("%.10f" % ((results_direct[i,j]-results_old[i,j])*100/results_old[i,j])),'%')
#        for j in range(0,len(SolarPanelSections)):
#            if abs(efficiencies_direct[j,i] - efficiencies_old[j,i]) > 1e-5:
#                print ('Efficiency NOT EQUAL',i,j)
#                print ('\t',efficiencies_direct[j,i],efficiencies_old[j,i])
#                print ('\tPercent Error:',("%.10f" % ((efficiencies_direct[j,i]-efficiencies_old[j,i])*100/efficiencies_old[j,i])),'%')
    print (time_direct,time_old)
#    for j in range(0,9):
#        plt.figure()
#        plt.plot(time,results_direct[:,j])
#        plt.plot(time,results_old[:,j])
#    print (SolarPowerMPP_Direct(SolarPanelSections))
#    print (SolarPowerMPP(SolarPanelSections))

#    best_guess = 0.5
#    lowest_time = 9.6
#    for k in range(0,101):
#        start = t.time()
#        Simulation_Direct(time,temperatures,fluxes,SolarPanelSections,I_pv_guess=k*0.01,high_efficiency=1.0,efficiency=1.0,UseAltaTempData=False)
#        end = t.time()
#        total = end-start
#        if total < lowest_time:
#            lowest_time = total
#            best_guess = k*0.01
#            print("lowest time:",lowest_time)
#            print("best guess:",best_guess)
"""

"""
#%% Validation Plots
if __name__ == "__main__":
    # Temperature Dependent Efficiency vs. Flux Calculations & Plots

    area = 60      # (m^2)
    voltage = 100  # (V)   [system open-circuit voltage]
    T = np.array([25,0,-25,-50,-75]) + 273.15  # (K)
    nlines = len(T)
    flux = range(0,1000) # (W/m^2)
    Model_efficiencies = np.zeros((len(flux),nlines))
    Alta_Devices_efficiencies = np.zeros((len(flux),nlines))

    section1 = Parameter(100,-51+273.15,1000,TotalMass=425,BattMass=212,SOC_initial=0.25,e_densityWhr=350,
                     Area_Desired=area,Voltage_Desired=voltage,N_s_guess=91,N_p_guess=770,I_pv=1,V_oc_single=1.1,
                     I_sc=0.23,T_coef_V=-0.187,T_coef_I=0.084,a=1.3,I_ph_ref=0.2310,R_p=623,R_s=0.1510,
                     cell_length=0.050,cell_width=0.0171,case_width=0.022)
    for i in range(0,len(flux)):
        section1.G = flux[i]
        for j in range(0,nlines):
            section1.T = T[j]
            Model_efficiencies[i][j] = SolarPowerMPP([section1],high_efficiency=1.0,efficiency=1.0)[8]
            Alta_Devices_efficiencies[i][j] = SolarPowerMPP([section1],high_efficiency=1.0,efficiency=1.0,UseAltaTempData=True)[8]

    plt.figure()
    plt.title("Efficiencies vs. Flux at " + str(int(voltage)) + " V (Model ––––  Alta Devices Data - - -)")
    plt.plot(flux,Model_efficiencies[:,0],'g-',label=str(int(T[0]-273.15))+" °C")
    plt.plot(flux,Model_efficiencies[:,1],'b-',label=str(int(T[1]-273.15))+" °C")
    plt.plot(flux,Model_efficiencies[:,2],'r-',label=str(int(T[2]-273.15))+" °C")
    plt.plot(flux,Model_efficiencies[:,3],'c-',label=str(int(T[3]-273.15))+" °C")
    plt.plot(flux,Model_efficiencies[:,4],'m-',label=str(int(T[4]-273.15))+" °C")
    plt.plot(flux,Alta_Devices_efficiencies[:,0],'g--')
    plt.plot(flux,Alta_Devices_efficiencies[:,1],'b--')
    plt.plot(flux,Alta_Devices_efficiencies[:,2],'r--')
    plt.plot(flux,Alta_Devices_efficiencies[:,3],'c--')
    plt.plot(flux,Alta_Devices_efficiencies[:,4],'m--')
    plt.xlim([0,len(flux)])
    plt.ylim([0.21,max(Model_efficiencies[:,4])+.01])
    plt.ylabel("Efficiency")
    plt.xlabel("Flux (W/m^2)")
    plt.legend(loc=4, ncol = 2)

    # Flux Dependent Current vs. Voltage Calculations & Plots
    # Solar Flux Variation  (Current vs. Voltage)
    area_cell = 0.0011  # (m^2) [area of a single cell]

    single_cell      = Parameter(V_pv=1.10,T=298.15,G=1000,Area_Desired=area_cell,Voltage_Desired=1.10,N_s_guess=1,N_p_guess=1)
    single_cell_2016 = Parameter(V_pv=1.09,T=298.15,G=1000,Area_Desired=area_cell,Voltage_Desired=1.09,N_s_guess=1,N_p_guess=1,
                                 V_oc_single=1.09,I_sc=0.246,I_ph_ref=0.2520,R_p=706)
    current_model2016 = np.zeros((400,4))
    current_model = np.zeros((400,4))
    current_data = np.zeros((400,4))
    voltages = np.linspace(0,2,400)
    for i in range(0,len(voltages)):
        single_cell.V_pv = voltages[i]
        single_cell_2016.V_pv = voltages[i]

        single_cell.G = 1000
        single_cell_2016.G = 1000
        current_model[i,0] = I_Single_String(single_cell,UseAltaTempData=False)*1000
        current_model2016[i,0] = I_Single_String(single_cell_2016,UseAltaTempData=False)*1000
        current_data[i,0] = I_Single_String(single_cell,UseAltaTempData=True)*1000

        single_cell.G = 500
        single_cell_2016.G = 500
        current_model[i,1] = I_Single_String(single_cell,UseAltaTempData=False)*1000
        current_model2016[i,1] = I_Single_String(single_cell_2016,UseAltaTempData=False)*1000
        current_data[i,1] = I_Single_String(single_cell,UseAltaTempData=True)*1000

        single_cell.G = 200
        single_cell_2016.G = 200
        current_model[i,2] = I_Single_String(single_cell,UseAltaTempData=False)*1000
        current_model2016[i,2] = I_Single_String(single_cell_2016,UseAltaTempData=False)*1000
        current_data[i,2] = I_Single_String(single_cell,UseAltaTempData=True)*1000

        single_cell.G = 100
        single_cell_2016.G = 100
        current_model[i,3] = I_Single_String(single_cell,UseAltaTempData=False)*1000
        current_model2016[i,3] = I_Single_String(single_cell_2016,UseAltaTempData=False)*1000
        current_data[i,3] = I_Single_String(single_cell,UseAltaTempData=True)*1000

    plt.figure()
    plt.plot(voltages, current_model2016[:,0], 'b-', label=r'$1000\ W/m^2$')
    plt.plot(voltages, current_model2016[:,1], 'r-', label=r'$500\ W/m^2$')
    plt.plot(voltages, current_model2016[:,2], 'g-', label=r'$200\ W/m^2$')
    plt.plot(voltages, current_model2016[:,3], 'k-', label=r'$100\ W/m^2$')
    plt.plot(voltages, current_model[:,0], 'b--')
    plt.plot(voltages, current_model[:,1], 'r--')
    plt.plot(voltages, current_model[:,2], 'g--')
    plt.plot(voltages, current_model[:,3], 'k--')
    plt.ylim(0,255)
    plt.yticks([50,100,150,200,250])
    plt.xlim(0,1.4)
    plt.ylabel('Current (mA)')
    plt.xlabel('Voltage (V)')
    plt.title('Single Cell Solar Flux Variation (Model: 2016 ––––  2017 - - -)')
    plt.legend(loc=1,prop={'size':9.5})
    plt.grid()

    plt.figure()
    plt.plot(voltages, current_model[:,0], 'b-')
    plt.plot(voltages, current_model[:,1], 'r-')
    plt.plot(voltages, current_model[:,2], 'g-')
    plt.plot(voltages, current_model[:,3], 'k-')
    plt.plot(voltages, current_data[:,0], 'b--', label=r'$1000\ W/m^2$')
    plt.plot(voltages, current_data[:,1], 'r--', label=r'$500\ W/m^2$')
    plt.plot(voltages, current_data[:,2], 'g--', label=r'$200\ W/m^2$')
    plt.plot(voltages, current_data[:,3], 'k--', label=r'$100\ W/m^2$')
    plt.ylim(0,255)
    plt.yticks([50,100,150,200,250])
    plt.xlim(0,1.4)
    plt.ylabel('Current (mA)')
    plt.xlabel('Voltage (V)')
    plt.title('Single Cell Solar Flux Variation (Model ––––  Alta Devices Data - - -)')
    plt.legend(loc=1,prop={'size':9.5})
    plt.grid()

    print ('Alta Devices Comparison Complete')

    #%% DC-DC efficiency effects
    start = t.time()
    temperatures = np.linspace(-80,0,8)+273.15
    efficiencies = np.empty((len(temperatures),5))
    for i in range(0,len(temperatures)):
        section1.T = temperatures[i]
        section1.G = 1000
        efficiencies[i,0]=SolarEfficiency([section1],efficiency=1.0)[0]
        section1.G = 600
        efficiencies[i,1]=SolarEfficiency([section1],efficiency=1.0)[0]
        section1.G = 400
        efficiencies[i,2]=SolarEfficiency([section1],efficiency=1.0)[0]
        section1.G = 200
        efficiencies[i,3]=SolarEfficiency([section1],efficiency=1.0)[0]
        section1.G = 100
        efficiencies[i,4]=SolarEfficiency([section1],efficiency=1.0)[0]
    end = t.time()
    print(end-start)

    plt.figure()
    plt.plot(temperatures,efficiencies[:,0],label='1000 W/m^2')
    plt.plot(temperatures,efficiencies[:,1],label='600 W/m^2')
    plt.plot(temperatures,efficiencies[:,2],label='400 W/m^2')
    plt.plot(temperatures,efficiencies[:,3],label='200 W/m^2')
    plt.plot(temperatures,efficiencies[:,4],label='100 W/m^2')
    plt.legend()

    # Efficiency vs flux at different temperatures
    start = t.time()
    fluxes = np.linspace(0,1000,1001)
    efficiencies1 = np.empty((len(fluxes),5,2))
    efficiencies2 = np.empty((len(fluxes),5,2))
    dcefficiency1=0.85
    dcefficiency2=1.0
    for i in range(0,len(fluxes)):
        section1.G = fluxes[i]
        section1.T = -75+273.15
        efficiencies1[i,0,:]=SolarEfficiency([section1],efficiency=dcefficiency1)
        efficiencies2[i,0,:]=SolarEfficiency([section1],efficiency=dcefficiency2)
        section1.T = -50+273.15
        efficiencies1[i,1,:]=SolarEfficiency([section1],efficiency=dcefficiency1)
        efficiencies2[i,1,:]=SolarEfficiency([section1],efficiency=dcefficiency2)
        section1.T = -25+273.15
        efficiencies1[i,2,:]=SolarEfficiency([section1],efficiency=dcefficiency1)
        efficiencies2[i,2,:]=SolarEfficiency([section1],efficiency=dcefficiency2)
        section1.T = 273.15
        efficiencies1[i,3,:]=SolarEfficiency([section1],efficiency=dcefficiency1)
        efficiencies2[i,3,:]=SolarEfficiency([section1],efficiency=dcefficiency2)
        section1.T = 25+273.15
        efficiencies1[i,4,:]=SolarEfficiency([section1],efficiency=dcefficiency1)
        efficiencies2[i,4,:]=SolarEfficiency([section1],efficiency=dcefficiency2)
    end = t.time()
    print(end-start)

    def EfficiencyPlot(efficiencies1,efficiencies2,ylab,dcefficiency1,dcefficiency2,yminn):
        plt.figure()
        plt.plot(fluxes,efficiencies1[:,4],'g',label='25°C')
        plt.plot(fluxes,efficiencies2[:,4],'g--')
        plt.plot(fluxes,efficiencies1[:,3],'b',label='0°C')
        plt.plot(fluxes,efficiencies2[:,3],'b--')
        plt.plot(fluxes,efficiencies1[:,2],'r',label='-25°C')
        plt.plot(fluxes,efficiencies2[:,2],'r--')
        plt.plot(fluxes,efficiencies1[:,1],'c',label='-50°C')
        plt.plot(fluxes,efficiencies2[:,1],'c--')
        plt.plot(fluxes,efficiencies1[:,0],'m',label='-75°C')
        plt.plot(fluxes,efficiencies2[:,0],'m--')
        plt.legend(loc=4,ncol=2,edgecolor='k')
        plt.xlim([0,1000])
        plt.axes().xaxis.set_minor_locator(MultipleLocator(50))
        plt.ylim([yminn,0.30])
        plt.xlabel("Flux (W/m^2)")
        plt.ylabel(ylab)
        plt.tick_params('y',left=True,labelleft=True,right=True,labelright=False,which='both')
        plt.title("Efficiencies vs. Flux (" + str(int(dcefficiency1*100)) + "% –––– " + str(int(dcefficiency2*100)) +"% - - - DC-DC Efficiency)")

    EfficiencyPlot(efficiencies1[:,:,1],efficiencies2[:,:,1],'System Efficiency (Solar & DC-DC Conversion)',dcefficiency1,dcefficiency2,0.15)
    EfficiencyPlot(efficiencies1[:,:,0],efficiencies2[:,:,1],'Solar Efficiency (No DC-DC Conversion Efficiency)',dcefficiency1,dcefficiency2,0.21)

    # Power vs Voltage Plots
    def PowerPlot(dcefficiencies,temp,flux):
        voltages=np.linspace(0,150,600)
        powers = np.zeros((len(voltages),6))
        section1.T = temp # [K]
        section1.G = flux # [W/m^2]
        for i in range(0,len(voltages)):
            powers[i,0] = SolarPowerWith_DC_DC_Conversion(voltages[i],1.0,dcefficiencies[0],[section1])[1]
            powers[i,1] = SolarPowerWith_DC_DC_Conversion(voltages[i],1.0,dcefficiencies[0],[section1])[0]
            powers[i,2] = SolarPowerWith_DC_DC_Conversion(voltages[i],1.0,dcefficiencies[1],[section1])[0]
            powers[i,3] = SolarPowerWith_DC_DC_Conversion(voltages[i],1.0,dcefficiencies[2],[section1])[0]
            powers[i,4] = SolarPowerWith_DC_DC_Conversion(voltages[i],1.0,dcefficiencies[3],[section1])[0]
            powers[i,5] = SolarPowerWith_DC_DC_Conversion(voltages[i],1.0,dcefficiencies[4],[section1])[0]
        plt.figure()
        plt.plot(voltages,powers[:,0],label='Before DC-DC Conversion')
        plt.plot(voltages[np.argmax(powers[:,0])],np.amax(powers[:,0]),'o',color='C0',ms=3)
        for i in range(1,6):
            plt.plot(voltages,powers[:,i],label=str(int(dcefficiencies[i-1]*100)) + '% DC-DC Conversion Efficiency')
            plt.plot(voltages[np.argmax(powers[:,i])],np.amax(powers[:,i]),'o',color='C'+str(i),ms=3)
        #plt.plot(self.solar_voltage[j],self.solar_power_to_system[j]/0.983,'o',ms=6,color='C1')
        plt.legend(fontsize=8)
        plt.xlim([0,120])
        plt.axis(ymin=0)
        plt.xlabel("Array Voltage (V)")
        plt.ylabel("Power (W)")
        plt.tick_params('y',left=True,labelleft=True,right=True,labelright=False,which='both')
        plt.title("Solar Array Power vs. Voltage at " + ("%.1f" % (temp-273.15)) + '°C and ' + ("%.1f" % flux) + ' W/m^2',fontsize=12)

    PowerPlot(np.array([99,97,95,90,85])/100.0,-75+273.15,1000)
    PowerPlot(np.array([99,97,95,90,85])/100.0,-50+273.15,1000)
    PowerPlot(np.array([99,97,95,90,85])/100.0,-25+273.15,1000)
    PowerPlot(np.array([99,97,95,90,85])/100.0,273.15,1000)
    PowerPlot(np.array([99,97,95,90,85])/100.0,25+273.15,1000)
    PowerPlot(np.array([99,97,95,90,85])/100.0,-56.49+273.15,1000)
    PowerPlot(np.array([99,97,95,90,85])/100.0,-56.49+273.15,750)
    PowerPlot(np.array([99,97,95,90,85])/100.0,-56.49+273.15,500)
    PowerPlot(np.array([99,97,95,90,85])/100.0,-56.49+273.15,300)
    PowerPlot(np.array([99,97,95,90,85])/100.0,-56.49+273.15,100)
    PowerPlot(np.array([99,97,95,90,85])/100.0,-56.49+273.15,50)
    PowerPlot(np.array([99,97,95,90,85])/100.0,-56.49+273.15,25)
    PowerPlot(np.array([99,97,95,90,85])/100.0,-56.49+273.15,10)


    PowerPlot(np.array([99,97,95,90,85])/100.0,-25+273.15,136)
    PowerPlot(np.array([99,97,95,90,85])/100.0,-25+273.15,139)
"""
