from scipy.optimize import fsolve
import numpy as np
from numpy import inf

def ApproximateModelParameters(Voc,Vmpp,Impp,Isc,Ncells,a,Rs_guess=0.1,Rp_guess=10,Iph_guess='none'):
    if type(Iph_guess) is str:
        Iph_guess = Isc
        
    ## Constants
    k=1.3806e-23 # Boltzman constant (J/oK)
    q=1.602e-19 # Elementary charge on an electron (oC)
    Tref=273.15+25 # K
    #Eg=1.12*q

    ## Calculations
    Vth = k*Tref/q # Thermal voltage
    Io=Isc/(np.exp(Voc/(a*Ncells*Vth))-1)

    ## Initialization
#    #Start conditions and increment value adjusted from previous run for best
#    #result
#    Rs_start=0.15
#    Rs_step=0.001
#    Rp_start=10
#    Rp_step=1
#    Iph_start=Isc
#    Iph_step=0.001
#    Min=inf
#
#    ## Iteration and minimization of error
#    for i in range(1,1000,1):
#        for j in range(1,1000,1):
#            for k in range(1,100,1):
#                    Rs=Rs_start+Rs_step*i
#                    Rp=Rp_start+Rp_step*j
#                    Iph=Iph_start+Iph_step*k
#                    Error1=(Vmpp/Impp)-((a*Ncells*Vth*Rp)/((Io*Rp*np.exp((Vmpp+(Impp*Rs))/(a*Ncells*Vth)))+(a*Ncells*Vth)))-Rs
#                    Error2=(Vmpp+(Impp*Rs))/(Iph-Impp-Io*(np.exp((Vmpp+(Impp*Rs))/(a*Ncells*Vth))-1))-Rp
#                    Error3=(((Rp+Rs)/Rp)*Isc)-Iph
#                    TotalError=Error1**2+Error2**2+Error3**2
#                    if TotalError<Min:
#                        #E1=Error1
#                        #E2=Error2
#                        #E3=Error3
#                        Min=TotalError
#                        Rsbest=Rs
#                        Rpbest=Rp
#                        Iphbest=Iph
#                        #I=i
#                        #J=j
#                        #K=k
#    ## Output Values
#    Rp=Rpbest
#    Rs=Rsbest
#    Iph=Iphbest
    
    def solver(X):
        Rs=abs(X[0])
        Rp=abs(X[1])
        Iph=abs(X[2])
        F1=(Vmpp/Impp)-((a*Ncells*Vth*Rp)/((Io*Rp*np.exp((Vmpp+(Impp*Rs))/(a*Ncells*Vth)))+(a*Ncells*Vth)))-Rs
        F2=(Vmpp+(Impp*Rs))/(Iph-Impp-Io*(np.exp((Vmpp+(Impp*Rs))/(a*Ncells*Vth))-1))-Rp
        F3=(((Rp+Rs)/Rp)*Isc)-Iph
        return F1,F2,F3
    Rs,Rp,Iph = abs(fsolve(solver,[Rs_guess,Rp_guess,Iph_guess]))
    print("\nRp = ",Rp)
    print("Rs = ",Rs)
    print("Iph = ",Iph)
    print (solver([Rs,Rp,Iph]))
    
    
#    if Rp < 0: # for troubleshooting
#        from mpl_toolkits.mplot3d import Axes3D
#        import matplotlib.pyplot as plt
#        Rp_array = np.linspace(0,200,1000)
#        Rs_array = np.linspace(0,1,1000)
#        F1_array = np.empty((len(Rp_array),len(Rs_array)))
#        F2_array = np.empty((len(Rp_array),len(Rs_array)))
#        F3_array = np.empty((len(Rp_array),len(Rs_array)))
##        for i in range(0,len(Rp_array)):
##            for j in range(0,len(Rs_array)):
##                F1_array[i,j] = (Vmpp/Impp)-((a*Ncells*Vth*Rp_array[i])/((Io*Rp_array[i]*np.exp((Vmpp+(Impp*Rs_array[j]))/(a*Ncells*Vth)))+(a*Ncells*Vth)))-Rs_array[j]
##                F2_array[i,j] = (Vmpp+(Impp*Rs_array[j]))/(Iph-Impp-Io*(np.exp((Vmpp+(Impp*Rs_array[j]))/(a*Ncells*Vth))-1))-Rp_array[i]
##                F3_array[i,j] = (((Rp_array[i]+Rs_array[j])/Rp_array[i])*Isc)-Iph
#        Rp_array,Rs_array = np.meshgrid(Rp_array,Rs_array)
#        F1_array = (Vmpp/Impp)-((a*Ncells*Vth*Rp_array)/((Io*Rp_array*np.exp((Vmpp+(Impp*Rs_array))/(a*Ncells*Vth)))+(a*Ncells*Vth)))-Rs_array
#        F2_array = (Vmpp+(Impp*Rs_array))/(Iph-Impp-Io*(np.exp((Vmpp+(Impp*Rs_array))/(a*Ncells*Vth))-1))-Rp_array
#        F3_array = (((Rp_array+Rs_array)/Rp_array)*Isc)-Iph
#        Ftot_array = F1_array**2 + F2_array**2 + F3_array**2
#        
#        for i in range(0,20):
#            plt.close(i)
#            
#        fig = plt.figure()
#        ax = fig.gca(projection='3d')
#        surf = ax.plot_surface(Rp_array,Rs_array,Ftot_array)
#        ax.set_zlim(-1,1)
#        
##        fig = plt.figure()
##        ax = fig.gca(projection='3d')
##        surf = ax.plot_surface(Rp_array,Rs_array,F1_array)
##        ax.set_zlim(-1,1)
##        
##        fig = plt.figure()
##        ax = fig.gca(projection='3d')
##        surf = ax.plot_surface(Rp_array,Rs_array,F2_array)
##        ax.set_zlim(-1,1)
##        
##        fig = plt.figure()
##        ax = fig.gca(projection='3d')
##        surf = ax.plot_surface(Rp_array,Rs_array,F3_array)
##        ax.set_zlim(-.25,.25)
    return Rp,Rs,Io,Iph

if __name__ == "__main__":
    ## Parameters from Alta Devices single cell (2016)
    #Voc=1.09 #V
    #Vmpp=0.96 #V
    #Impp=0.236 #A
    #Isc=0.246 #A
    #Ncells=1
    #a=1.3
    
    # Parameters from Alta Devices single cell (2017)
    Voc=1.1 #V
    Vmpp=0.96 #V
    Impp=0.22 #A
    Isc=0.23 #A
    Ncells=1
    a=1.3
    print ('\nAlta:')
    [Rp_Alta,Rs_Alta,Io_Alta,Iph_Alta]=ApproximateModelParameters(Voc,Vmpp,Impp,Isc,Ncells,a)
    
    # Parameters from SunPower C60 Solar Cell (Nov 2010) - Bin G
    Voc=0.682
    Vmpp=0.574
    Impp=5.83
    Isc=6.24
    Ncells=1
    a=1.3
    print ('\nG:')
    [Rp_G,Rs_G,Io_G,Iph_G]=ApproximateModelParameters(Voc,Vmpp,Impp,Isc,Ncells,a,Rs_guess=0.001,Rp_guess=10)
    
    # Parameters from SunPower C60 Solar Cell (Nov 2010) - Bin H
    Voc=0.684
    Vmpp=0.577
    Impp=5.87
    Isc=6.26
    Ncells=1
    a=1.3
    print ('\nH:')
    [Rp_H,Rs_H,Io_H,Iph_H]=ApproximateModelParameters(Voc,Vmpp,Impp,Isc,Ncells,a,Rs_guess=0.001,Rp_guess=10)
    
    # Parameters from SunPower C60 Solar Cell (Nov 2010) - Bin I
    Voc=0.686
    Vmpp=0.581
    Impp=5.90
    Isc=6.27
    Ncells=1
    a=1.3
    print ('\nI:')
    [Rp_I,Rs_I,Io_I,Iph_I]=ApproximateModelParameters(Voc,Vmpp,Impp,Isc,Ncells,a,Rs_guess=0.001,Rp_guess=10)
    
    # Parameters from SunPower C60 Solar Cell (Nov 2010) - Bin J
    Voc=0.687
    Vmpp=0.582
    Impp=5.93
    Isc=6.28
    Ncells=1
    a=1.3
    print ('\nJ:')
    [Rp_J,Rs_J,Io_J,Iph_J]=ApproximateModelParameters(Voc,Vmpp,Impp,Isc,Ncells,a,Rs_guess=0.001,Rp_guess=100) 
    
#    # Parameters from SunPower SPR-E-Flex-110 Solar Array (Feb 2017)
#    Voc=21.7
#    Vmpp=18.5
#    Impp=6.0
#    Isc=6.3
#    Ncells=32
#    a=1.3
#    print ('\n110:')
#    [Rp_110,Rs_110,Io_110,Iph_110]=ApproximateModelParameters(Voc,Vmpp,Impp,Isc,Ncells,a,Rs_guess=0.06,Rp_guess=600)
#    
#    # Parameters from SunPower SPR-E-Flex-100 Solar Array (Feb 2017)
#    Voc=21
#    Vmpp=17.5
#    Impp=5.8
#    Isc=6.2
#    Ncells=32
#    a=1.3
#    print ('\n100:')
#    [Rp_100,Rs_100,Io_100,Iph_100]=ApproximateModelParameters(Voc,Vmpp,Impp,Isc,Ncells,a,Rs_guess=0.07,Rp_guess=450)
#
#    # Parameters from SP 144
#    Voc=30.0
#    Vmpp=25.3
#    Impp=5.7
#    Isc=6.0
#    Ncells=44
#    a=1.3
#    print ('\n144:')
#    [Rp144,Rs144,Io144,Iph144]=ApproximateModelParameters(Voc,Vmpp,Impp,Isc,Ncells,a,Rs_guess=0.04,Rp_guess=50)
#    
#    # Parameters from SP 130
#    Voc=27.3
#    Vmpp=22.8
#    Impp=5.7
#    Isc=6.0
#    Ncells=40
#    a=1.3
#    print ('\n130:')
#    [Rp130,Rs130,Io130,Iph130]=ApproximateModelParameters(Voc,Vmpp,Impp,Isc,Ncells,a,Rs_guess=0.001,Rp_guess=10)
#    
#    # Parameters from SP 118 L & Q
#    Voc=24.5
#    Vmpp=20.7
#    Impp=5.7
#    Isc=6.0
#    Ncells=36
#    a=1.3
#    print ('\n118:')
#    [Rp118,Rs118,Io118,Iph118]=ApproximateModelParameters(Voc,Vmpp,Impp,Isc,Ncells,a,Rs_guess=0.001,Rp_guess=10)
#    
#    # Parameters from SP 104
#    Voc=21.8
#    Vmpp=18.2
#    Impp=5.7
#    Isc=6.0
#    Ncells=32
#    a=1.3
#    print ('\n104:')
#    [Rp104,Rs104,Io104,Iph104]=ApproximateModelParameters(Voc,Vmpp,Impp,Isc,Ncells,a,Rs_guess=0.001,Rp_guess=10)
#    
#    # Parameters from SP 78
#    Voc=16.4
#    Vmpp=13.7
#    Impp=5.7
#    Isc=6.0
#    Ncells=24
#    a=1.3
#    print ('\n78:')
#    [Rp78,Rs78,Io78,Iph78]=ApproximateModelParameters(Voc,Vmpp,Impp,Isc,Ncells,a,Rs_guess=0.001,Rp_guess=10)
#    
#    # Parameters from SP 52 L
#    Voc=10.9
#    Vmpp=9.1
#    Impp=5.7
#    Isc=6.0
#    Ncells=16
#    a=1.3
#    print ('\n52L:')
#    [Rp52L,Rs52L,Io52L,Iph52L]=ApproximateModelParameters(Voc,Vmpp,Impp,Isc,Ncells,a,Rs_guess=0.03,Rp_guess=1200)
