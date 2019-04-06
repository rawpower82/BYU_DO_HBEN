# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 12:52:21 2017

@author: hunterrawson
"""

import numpy as np
import pandas as pd
from scipy.optimize import fsolve,brent
from ApproximateSolarModelParameters import ApproximateModelParameters
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter

def Series_vs_Parallel_Modules(area,voltage,Nm_s_guess=8,Nm_p_guess=4,V_oc_single=1.1,cells_per_module=96,
                             cell_width=0.125,cell_diagonal=0.160,module_width=41.5*0.0254,module_length=62.6*0.0254):
    '''
    # Calculates number of solar modules to have in series vs. in parallel
    # area - (m^2) [Total area of solar array]
    # voltage - (V) [Desired open circuit voltage of solar array]
    # Nm_s_guess - [Guess for number of modules in series]
    # Nm_p_guess - [Guess for number of modules in parallel]
    # V_oc_single - (V/cell) [Open Circuit Voltage of Single Cell]
    # cells_per_module - [number of solar cells in a single module]
    # cell_width - (m) [width of a single solar cell]
    # cell_diagonal - (m) [length of diagonal from one corner of the cell to the other]
    # module_width - (m) [width of a single solar cell aperture]
    # module_length - (m) [length of a module]
    '''
    def functions(X):
        N_s = X[0]
        N_p = X[1]
        F = [0,0]
        F[0] = area - N_s*N_p*module_width*module_length
        F[1] = voltage - V_oc_single*N_s
        return F

    NmsNmp_guess = [Nm_s_guess,Nm_p_guess]
    NmsNmp = fsolve(functions,NmsNmp_guess)
    Nm_s = int(NmsNmp[0]+0.5)  # [Number of solar cells in series] (+0.5 rounds up value)
    Nm_p = int(NmsNmp[1]+0.5)  # [Number of strings of solar cells in parallel]

    SolarPanelArea = Nm_s*Nm_p*module_width*module_length # (m^2) (solar array surface area)
    VoltageOpenCircuit = V_oc_single*Nm_s          # (V) [open circuit voltage of the array]

    while (SolarPanelArea > area): # loop that ensures Solar Panel area is less than "area" rather than greater
        Nm_p -= 1
        SolarPanelArea = Nm_s*Nm_p*module_width*module_length

    SingleCellArea = cell_width**2 # (m^2) [single solar cell surface area (aperture area not included)
    # Area correction for corners of Cells
    diagonal_square = (2*cell_width**2)**(1/2) # m
    diagonal_small_square = diagonal_square - cell_diagonal # m - the diagonal length of a small square (two corners put together)
    length_small_square = (diagonal_small_square**2/2)**(1/2) # m - the side length of a small square (two corners put together)
    area_small_squares = 2*length_small_square**2 # Area of two small squares (4 small triangles) removed from each cell's area
    SingleCellArea -= area_small_squares # m^2 - remove area for each cell's corners
    SolarCellArea = SingleCellArea*cells_per_module*Nm_s*Nm_p # (m^2) [solar cell surface area (aperture area not included - used for efficiency calculations)]
    return [Nm_s,Nm_p,SolarPanelArea,SolarCellArea,VoltageOpenCircuit]

class Parameter:
    '''
    Definition of a parameter class, which is used to simplify function parameters
    Default parameters from Panasonic Photovoltaic module HIT (VBHN330SA16)
    '''
    def __init__(self,V_pv=500,T=25+273.15,G=1000,Area_Desired=1500*0.092903,Voltage_Desired=557.6,Nm_s_guess=8,Nm_p_guess=4,I_pv='none',
                 V_oc=69.7,I_sc=6.07,Vmpp=58.0,Impp=5.70,Ncells=8*12,K_V=-0.174,K_I=1.82e-3,a=1.3,
                 cell_width=0.125,cell_diagonal=0.160,module_width=41.457*0.0254,module_length=62.598*0.0254,Print=False):
        self.V_pv = V_pv                   # (V) [Voltage panel is operating at - "voltage output of the series coupling" (Vika)]
        self.T = T                         # (K) [temperature of solar panels]
        self.G = G                         # (W/m^2) [solar flux]
        [self.Nm_s,self.Nm_p,self.SolarPanelArea,self.SolarCellArea,self.SolarVoltageOpenCircuit] = Series_vs_Parallel_Modules(Area_Desired,
                                     Voltage_Desired,Nm_s_guess,Nm_p_guess,V_oc,Ncells,cell_width,cell_diagonal,module_width,module_length)
            # Nm_s - [Number of solar cells in series]
            # Nm_p - [Number of strings of solar cells in parallel]
            # SolarPanelArea - (m^2) [solar array surface area]
            # SolarCellArea - (m^2) [solar cell surface area (aperture area not included - used for efficiency calculations)]
            # SolarVoltageOpenCircuit - (V) [open circuit voltage of solar panels in series]
        self.N_s = self.Nm_s*Ncells        # [Number of solar cells in series]
        self.N_p = self.Nm_p               # [Number of strings of solar cells in parallel]
        self.V_oc_module = V_oc            # (V/module) [Open Circuit Voltage of a Single Module]
        self.V_oc_single = V_oc/Ncells     # (V/cell) [Open Circuit Voltage of Single Cell]
        self.I_sc = I_sc                   # (A) [Short Circuit Current of Single Cell]
        self.Vmpp = Vmpp                   # (V) [Maximum power point voltage]
        self.Impp = Impp                   # (A) [Maximum power point current]
        self.K_V = K_V                     # (V/K)
        self.K_I = K_I                     # (A/K)
        self.a = a                         # [Ideality Factor]
        self.R_p,self.R_s,Io,self.I_ph_ref = ApproximateModelParameters(V_oc/Ncells,Vmpp/Ncells,Impp,I_sc,1,a,Rs_guess=0.001,Rp_guess=10,Iph_guess='none',Print=Print)
            # (Ohms) {Equivalent parallel resistance of a single cell]
            # (Ohms) [Equivalent series resistance of a single cell]
            # (A) [Photo Current of a single cell at 25°C and 1000 W/m^2]
        if type(I_pv) is str:
            self.I_pv = self.I_ph_ref # (A) [Initial guess for current of cells in series]
        else:
            self.I_pv = I_pv                   # (A) [Initial guess for current of cells in series]


#HIT = Parameter(Area_Desired=1500*.092903,Voltage_Desired=557.6,Print=True)
#print (HIT.N_s)
#print (HIT.N_p)
#print (HIT.SolarPanelArea)
#print (HIT.SolarCellArea)

def I_Single_String(parameters,PrintF=False,ReturnNegative=False):
    '''
    # parameters - Parameter object
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
        K_V = parameters.K_V*parameters.Nm_s      # (V/K) [scales with number of modules in series]
        K_I = parameters.K_I                      # (A/K)
        a = parameters.a                          # [Ideality Factor]
        R_p = parameters.R_p                      # (Ohms) {Equivalent parallel resistance of a single cell]
        R_s = parameters.R_s                      # (Ohms) [Equivalent series resistance of a single cell]
        I_ph_ref = parameters.I_ph_ref            # (A) [Photo Current of a single cell at 25°C and 1000 W/m^2]

        R_p = N_s*R_p # (Ohms) [Equivalent parallel resistance of solar cells in series]
        R_s = N_s*R_s # (Ohms) [Equivalent series resistance of solar cells in series]

        q = 1.60217646e-19 # (C) [Charge of an electron]
        k_B = 1.38064852e-23 # (J/K) [Boltzmann constant]
        T_n = 25 + 273.15 # (K)
        Delta_T = T-T_n
        G_n = 1000 # (W/m^2)

        V_t = N_s*k_B*T/q # N_s*k_B*T/q (V) [Thermal voltage of solar cells in series (Vika page 22)]


        I_ph = (I_ph_ref + K_I*Delta_T)*(G/G_n) # (A) [Photo Current of solar cells in series - accounts for the difference in temperature and flux from 25°C and 1000 W/m^2]
        I_0 = (I_sc + K_I*Delta_T)/(np.exp((V_oc+K_V*Delta_T)/(a*V_t))-1) # (A)
        #print (I_0,I_sc,V_oc,a,V_t,(np.exp((V_oc+K_V*Delta_T)/(a*V_t))-1))
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
                #print ('negative',I_pv)
                I_pv = 0
    return I_pv
    # Assumptions:
        #  All Identical cells in series
        #  Temperature and irradiance are identical for entire block of series connected cells
            # "In series connected solar cells, the current for the chain is set by the current of the worst performing cell."

def SolarPower_Solve(V_pv,parameters_group):
    '''
    # V_pv - (V) [Voltage panel is operating at]
    # parameters_group - [array containing several Parameter objects]
    '''
    num_divisions = len(parameters_group)
    voltage_in = V_pv # (V) [voltage entering the inverter]
    current = 0 # (A) [Current entering the inverter from entire solar array]
    for i in range(0,num_divisions):
        parameters_group[i].V_pv = V_pv
        if parameters_group[i].G > 0: # Current will be 0 if flux is negative or 0 - time saver
            current += I_Single_String(parameters_group[i])*parameters_group[i].N_p
    power = voltage_in*current         # (W) [power entering the inverter]
    return -power

def SolarPower(V_pv,parameters_group):
    '''
    # V_pv - (V) [Voltage panel is operating at]
    # parameters_group - [array containing several Parameter objects]
    '''
    num_divisions = len(parameters_group)
    solar_section_efficiencies = np.zeros(num_divisions)      # (array, unitless) [efficiencies of each solar section]
    voltage_in = V_pv                                         # (V) [voltage entering the inverter]
    current = 0                                            # (A) [Current entering the inverter from entire solar array]
    for i in range(0,num_divisions):
        parameters_group[i].V_pv = V_pv
        if parameters_group[i].G > 0: # Current will be 0 if flux is negative or 0 - time saver
            current_single = I_Single_String(parameters_group[i])*parameters_group[i].N_p
            current += current_single
            solar_section_efficiencies[i] = voltage_in*current_single/(parameters_group[i].G*parameters_group[i].SolarCellArea)
        if parameters_group[0].SolarVoltageOpenCircuit != parameters_group[i].SolarVoltageOpenCircuit:
            print ("Error! Not all strings of solar cells are configured at the same voltage!")
    power = voltage_in*current         # (W) [power entering the inverter]
    return power, current, solar_section_efficiencies

def SolarPowerMPP(parameters_group):
    '''
    # parameters_group - [array containing several Parameter objects]
    '''
    num_divisions = len(parameters_group)
    total_flux = 0                                  # (W/m^2) [Flux hitting the entire array]
    total_flux_power = 0                            # (W) [Flux power hitting the entire array]
    for i in range(0,num_divisions):
        total_flux += parameters_group[i].G
        total_flux_power += parameters_group[i].G * parameters_group[i].SolarCellArea

    args = (parameters_group,)
    if total_flux <= 0:
        Vopt = 0
        solar_efficiency = 0
        [power,current,solar_section_efficiencies] = [0,0,np.zeros(num_divisions)]
    else:
        Vopt = brent(SolarPower_Solve, args=args, brack=[parameters_group[0].SolarVoltageOpenCircuit*2,1e-12],
                     tol=1.48e-8, full_output=0, maxiter=500) # This uses Brent's Method for optimization
                     # (V) [Optimal operating voltage of the solar array]
        [power,current,solar_section_efficiencies] = SolarPower(Vopt,parameters_group)
        solar_efficiency = power/total_flux_power
    return power, current, Vopt, solar_efficiency, solar_section_efficiencies
    '''
    # power - (W) [Power entering the inverter from the solar array]
    # current - (A) [Current entering the inverter from entire solar array]
    # Vopt - (V) [Optimal operating voltage of the solar array]
    # solar_efficiency - (unitless) [Efficiency of solar panels]
    # solar_section_efficiencies - (array, unitless) [efficiencies of each solar section]
    '''

def OrientationCorrection(DirectFlux,Azimuth,Zenith,RoofDirection,RoofPitch=26.6,ViewPlot=False):
        '''
        DirectFlux    [float,W/m^2]   (Direct tracking flux)
        Azimuth       [float,degrees] (Sun's azimuth angle clockwise from north)
        Zenith        [float,degrees] (Sun's zenith angle; 0 = up, 90 = horizon)
        RoofDirection [float,degrees] (Roof spine angle clockwise from north)
                                      ( = 0 for N-S; = 90 for E-W)
        RoofPitch     [float,degrees] (Roof pitch angle)
        '''
        A = np.radians(-Azimuth + 90) # Convert azimuth angle to standard math
        # coordinates (x-axis = 0 , positive counter-clockwise)
        Z = np.radians(Zenith)

        # Convert sun angles to vector
        u = np.cos(A)*np.sin(Z) # east
        v = np.sin(A)*np.sin(Z) # north
        w = np.cos(Z) # up
        sun_norm = np.sqrt(u**2+v**2+w**2)

        # Calculate surface normal
        c1_l = np.cos(np.radians(RoofPitch))    # Roof pitch
        c1_r = np.cos(-np.radians(RoofPitch))     # Roof pitch
        #c2 = np.cos(0.0)                         # Spine is parallel to ground
        c3 = np.cos(np.radians(RoofDirection))   # Spine angle clockwise from north
        s1_l = np.sin(np.radians(RoofPitch))    # Roof pitch
        s1_r = np.sin(-np.radians(RoofPitch))     # Roof pitch
        #s2 = np.sin(0.0)                         # Spine is parallel to ground
        s3 = np.sin(np.radians(RoofDirection))   # Spine angle clockwise from north

        n1_l = -c3*s1_l #c1*s2*s3 - c3*s1 # east - s2 = 0
        n1_r = -c3*s1_r #c1*s2*s3 - c3*s1 # east - s2 = 0
        n2_l = s1_l*s3 #c1*c3*s2 + s1*s3 # north - s2 = 0
        n2_r = s1_r*s3 #c1*c3*s2 + s1*s3 # north - s2 = 0
        n3_l = c1_l #c1*c2 # up - c2 = 1
        n3_r = c1_r #c1*c2 # up - c2 = 1
        norm = np.sqrt(n1_l**2+n2_l**2+n3_l**2)

        # Obliquity factor (0-1)
        mu_l = u/sun_norm*n1_l/norm + v/sun_norm*n2_l/norm + w/sun_norm*n3_l/norm
        mu_r = u/sun_norm*n1_r/norm + v/sun_norm*n2_r/norm + w/sun_norm*n3_r/norm

        # Clip mu to >= 0, (goes negative if module is facing away from sun)
        if(mu_l < 0):
            mu_l = 0
        if(mu_r < 0):
            mu_r = 0

        # Local flux for each side
        LocalFluxL = mu_l * DirectFlux # (W/m^2)
        LocalFluxR = mu_r * DirectFlux # (W/m^2)

        if ViewPlot:
            fig = plt.figure()
            ax = Axes3D(fig)
            plt.quiver(np.array([0]),np.array([0]),np.array([0]),[u/sun_norm],[v/sun_norm],[w/sun_norm],normalize=False,color='red')
            plt.quiver(np.array([0]),np.array([0]),np.array([0]),[n1_l/norm],[n2_l/norm],[n3_l/norm],normalize=False,color='grey')
            plt.quiver(np.array([0]),np.array([0]),np.array([0]),[n1_r/norm],[n2_r/norm],[n3_r/norm],normalize=False,color='black')
            plt.plot([0],[0],[0],'ko')
            plt.plot([u/sun_norm],[v/sun_norm],[w/sun_norm],'ro')
            ax.set_xlim([-1,1])
            ax.set_ylim([-1,1])
            ax.set_zlim([0,1])
            ax.set_xlabel('e')
            ax.set_ylabel('n')
            ax.set_zlabel('u')

        return LocalFluxL,LocalFluxR,mu_l,mu_r

def PrintTime(Hour,minute=True,second=False):
    if 12 <= Hour < 24:
        am_pm = 'PM'
    else:
        am_pm = 'AM'
    if 13 <= Hour < 25:
        Hour -= 12
    elif 25 <= Hour:
        Hour -= 24
    if second == False:
        if minute == False:
            statement = str("%d" % int(Hour)) + ' ' + am_pm
        else:
            statement = str("%d:%02d" % (int(Hour),(Hour*60) % 60)) + ' ' + am_pm
    else:
        statement = str("%d:%02d:%02d" % (int(Hour),(Hour*60) % 60,(Hour*3600) % 60)) + ' ' + am_pm
    return statement

if __name__ == "__main__":
    HIT_Single = Parameter(Area_Desired=1.7,Voltage_Desired=70)
    plt.close('all')
    #%% Current vs. Voltage
    n = 100
    Vs = np.linspace(0,80,n)
    Ts = np.ones(n)*298.15 # (K) [cell temperature]
    Gs = np.empty(((n,5)))
    Is = np.empty((n,5))
    for j in range(5):
        Gs[:,j] = np.ones(n)*1000-200*j # (W/m^2) [solar flux]
    for i in range(n):
        HIT_Single.V_pv = Vs[i]
        HIT_Single.T = Ts[i]
        for j in range(5):
            HIT_Single.G = Gs[i,j]
            Is[i,j] = I_Single_String(HIT_Single,ReturnNegative=True)

    plt.figure(figsize=(7,6))
    plt.plot(Vs, Is[:,0], '-', color='darkorange', label=r'$1000\ W/m^2$', linewidth=2)
    plt.plot(Vs, Is[:,1], '-', color='darkorange', label=r'$800\ W/m^2$', linewidth=1)
    plt.plot(Vs, Is[:,2], '-', color='darkorange', label=r'$600\ W/m^2$', linewidth=1)
    plt.plot(Vs, Is[:,3], '-', color='darkorange', label=r'$400\ W/m^2$', linewidth=1)
    plt.plot(Vs, Is[:,4], '-', color='darkorange', label=r'$200\ W/m^2$', linewidth=1)
    plt.ylim(0,7.0)
    plt.yticks(np.linspace(0,7,8))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xlim(0,80)
    plt.xticks(np.linspace(0,80,9))
    plt.ylabel('Current (A)')
    plt.xlabel('Voltage (V)')
    plt.legend(loc=1,prop={'size':9.5})
    plt.grid()

    #%% Change in Irradiance
    df = pd.read_csv('history_export_2019-03-21T21_33_39.csv')
    time = df['Hour'][216:239].values
    Ts = (df['Temperature  [2 m above gnd]'][216:239].values-32)*5/9+273.15 # K
    Gs = df['Shortwave Radiation  [sfc]'][216:239].values # W/m2
    n = len(time)
    #time = np.linspace(7,17,n)
    #Ts = -70*((time-12)/12)**2 + 305
    #Gs = -8000*((time-12)/12)**2 + 1200
    #Gs[Gs < 0] = 0.0
    Ps = np.empty(n)
    Is = np.empty(n)
    Vs = np.empty(n)
    etas = np.empty(n)
    for i in range(n):
        HIT_Single.G = Gs[i]
        HIT_Single.T = Ts[i]
        results = SolarPowerMPP([HIT_Single])
        Ps[i],Is[i],Vs[i],etas[i],null = results

    xtix = np.arange(int(time[0]),int(time[-1]+1),2)
    xtixnames = []
    xtixblank = []
    for i in range(0,len(xtix)):
        xtixnames.append(PrintTime(xtix[i],minute=False))
        xtixblank.append('')

    plt.figure(figsize=(10,16/3))
    plt.subplot(2,1,1)
    plt.plot(time,Ts-273.15,color='darkorange')
    plt.ylabel('Temperature (°C)')
    plt.xlim([time[0],time[-1]])
    plt.xticks(xtix,xtixblank)

    plt.subplot(2,1,2)
    plt.plot(time,Gs,color='darkorange')
    plt.ylabel('Solar Irradiance (W/m^2)')
    plt.xlim([time[0],time[-1]])
    plt.xticks(xtix,xtixblank)

    #plt.subplot(3,1,3)
    #plt.plot(time,etas*100,color='darkorange')
    #plt.xlabel('Time (hr)')
    #plt.ylabel('Solar Efficiency (%)')
    #plt.xlim([time[0],time[-1]])
    #plt.xticks(xtix,xtixnames)

    plt.figure(figsize=(10,8))
    plt.subplot(3,1,1)
    plt.plot(time,Ps,color='darkorange')
    plt.ylabel('Power (W)')
    plt.xlim([time[0],time[-1]])
    plt.xticks(xtix,xtixblank)

    plt.subplot(3,1,2)
    plt.plot(time,Is,color='darkorange')
    plt.ylabel('Current (A)')
    plt.xlim([time[0],time[-1]])
    plt.xticks(xtix,xtixblank)

    plt.subplot(3,1,3)
    plt.plot(time,Vs,color='darkorange')
    plt.xlabel('Time (hr)')
    plt.ylabel('Voltage (V)')
    plt.xlim([time[0],time[-1]])
    plt.xticks(xtix,xtixnames)

    #%%
    DirectFlux = 1000 # W/m2
    Azimuth = 60
    Zenith = 15
    RoofDirection = 0 # N-S; 90 for E-W
    RoofPitch = 26.6
    LocalFluxL,LocalFluxR,mu_l,mu_r = OrientationCorrection(DirectFlux,Azimuth,Zenith,
                                                            RoofDirection,RoofPitch,ViewPlot=True)