import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# define energy balance model
def Battery(x,t):
    
    #Tesla Car Battery
    D2=30.0    # [kW] max continuous discharge rate
    C2=60.0
    Q2=87.0   # [kWh] storage capacity  (77)
    Q2_kJ=Q2*3600.0
    
    # Charging and Discharging Parameters
    # Fractional discharging rate 
    dis1=[-1.00e-29, 9.12e-03, 1.593965, 1.0]

    # Fractional charging rate 
    char1=[1.0, 1.0+1e-30, -9.11e-03, -1.593965303]
    
    # Tesla Home Battery
    D1=5.0    # [kW] max continuous discharge rate
    C1=15.0    # [kW] max continuous charge rate
    Q1=13.5    # [kWh] storage capacity
    NN_PW=1.0  # Number of powerwall batteries;
    Q1_kJ=Q1*NN_PW*3600.0
    
    # Parameters
    if x[0]>1.0:
        x[0]=1.0
    if x[1]>1.0:
        x[1]=1.0
    discharge1_max=D1*(dis1[3]/(1+(dis1[1]/(x[0]-dis1[0]))**dis1[2]))
    charge1_max=C1*(1/(char1[0]+((x[0]-char1[1])/char1[2])**char1[3]))
    
    # Battery #2 use the same charge discharge rate as a function of their
    # own SOC
    discharge2_max=D2*(dis1[3]/(1+(dis1[1]/(x[1]-dis1[0]))**dis1[2]))
    charge2_max=C2*(1/(char1[0]+((x[1]-char1[1])/char1[2])**char1[3]))
    
    if power>0.0: # +ve is need
        discharge_kW=power
        charge_kW=0.0
    elif power<0.0:
        discharge_kW=0.0
        charge_kW=-power
    elif power==0.0:
        discharge_kW=0.0
        charge_kW=0.0
    
    # Use home battery first
    if discharge_kW>discharge1_max:
        discharge2_kW=discharge_kW-discharge1_max
        discharge1_kW=discharge1_max        
        if discharge2_kW>discharge2_max:
            discharge2_kW=discharge2_max
    else:
        discharge2_kW=0.0
        discharge1_kW=discharge_kW

    # Charge car battery first
    if charge_kW>charge2_max:
        charge1_kW=charge_kW-charge2_max
        charge2_kW=charge2_max        
        if charge1_kW>charge1_max:
            charge1_kW=charge1_max
    else:
        charge1_kW=0.0
        charge2_kW=charge_kW

    # Energy Balances
    dy1dt=(charge1_kW-discharge1_kW)/Q1_kJ; 
    dy2dt=(charge2_kW-discharge2_kW)/Q2_kJ; 
    
    return np.array([dy1dt,dy2dt])

n = 60*300+1  # Number of second time points (10min)

# Power Required (kW)
P = np.ones(n)
# Battery Model Input
Use = np.zeros(n)
Store = np.zeros(n)
Use[1:]=5.0
Use[15*60+1:]=30.0
Use[30*60+1:]=55.0
Use[50*60+1:]=0.0
Use[60*60+1:]=0.0
Use[100*60+1:]=3.0
Use[160*60+1:]=10.0
Use[190*60+1:]=0.0

Store[1:]=10.0
Store[15*60+1:]=0.0
Store[30*60+1:]=64.0
Store[50*60+1:]=64.0
Store[60*60+1:]=20.0
Store[100*60+1:]=0.0
Store[160*60+1:]=0.0
Store[190*60+1:]=20.0

# +ve P means the house is requesting power
# -ve P means the house has energy to store (i.e. solar panels or cheap grid elec)
P = Use - Store

# Initial Conditions - SOC
SOC10=0.5
SOC20=0.5
# Initialize SOCs
SOC1 = np.ones(n)*SOC10
SOC2 = np.ones(n)*SOC20
usage_kW_v = np.zeros(n)
storage_kW_v = np.zeros(n)
time = np.linspace(0,n-1,n) # Time vector
x0=np.zeros(2)
for i in range(1,n):
    
    if P[i-1]>0:
        usage_kW_v[i]=P[i-1]
        storage_kW_v[i]=0.0
    else:
        usage_kW_v[i]=0.0
        storage_kW_v[i]=-P[i-1]
    # initial condition for next step    
    x0[0]=SOC1[i-1]
    x0[1]=SOC2[i-1]
    
    # time interval for next step
    tm = [time[i-1],time[i]]
    
    # input power for next step
    power = (P[i-1])

    # Integrate ODE for 1 sec each loop
    x = odeint(Battery,x0,tm)
    
    #record SOC values for both batteries
    SOC1[i] = x[-1][0]
    SOC2[i] = x[-1][1]
#    print("i = ", i, "SOC1 = ", SOC1[i], "SOC2 = ", SOC2[i], "Power = ", power)

# Plot results
plt.figure(1)
plt.subplot(2,1,1)
# plt.plot(time/3660.0,P,'g--',label=r'$Power$', linewidth = 1.5)
plt.plot(time/3660.0,usage_kW_v ,'g--',label=r'$Usage$', linewidth = 1.5)
plt.plot(time/3660.0,storage_kW_v,'b--',label=r'$Storage$', linewidth = 1.5)
plt.ylabel('Power (%)')
plt.legend(loc='best')
plt.xlabel('Time (hr)')
plt.show()

plt.subplot(2,1,2)
plt.plot(time/3660.0,SOC1*100.0,'g--',label=r'$Powerwall$', linewidth = 1.5)
plt.plot(time/3660.0,SOC2*100.0,'b--',label=r'$Model S$', linewidth = 1.5)
plt.ylabel('SOC (%)')
plt.legend(loc='best')

plt.xlabel('Time (hr)')
plt.show()
