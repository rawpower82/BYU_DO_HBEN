from scipy.optimize import fsolve
import numpy as np

def ApproximateModelParameters(Voc,Vmpp,Impp,Isc,Ncells,a,Rs_guess=0.1,Rp_guess=10,Iph_guess='none',Print=False):
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

    def solver(X):
        Rs=abs(X[0])
        Rp=abs(X[1])
        Iph=abs(X[2])
        F1=(Vmpp/Impp)-((a*Ncells*Vth*Rp)/((Io*Rp*np.exp((Vmpp+(Impp*Rs))/(a*Ncells*Vth)))+(a*Ncells*Vth)))-Rs
        F2=(Vmpp+(Impp*Rs))/(Iph-Impp-Io*(np.exp((Vmpp+(Impp*Rs))/(a*Ncells*Vth))-1))-Rp
        F3=(((Rp+Rs)/Rp)*Isc)-Iph
        return F1,F2,F3
    Rs,Rp,Iph = abs(fsolve(solver,[Rs_guess,Rp_guess,Iph_guess]))
    if Print == True:
        print("\nRp = ",Rp)
        print("Rs = ",Rs)
        print("Iph = ",Iph)
        print (solver([Rs,Rp,Iph]))

    return Rp,Rs,Io,Iph

if __name__ == "__main__":
    # Parameters from Alta Devices single cell (2017)
    Voc=1.1 #V
    Vmpp=0.96 #V
    Impp=0.22 #A
    Isc=0.23 #A
    Ncells=1
    a=1.3
    print ('\nAlta:')
    [Rp_Alta,Rs_Alta,Io_Alta,Iph_Alta]=ApproximateModelParameters(Voc,Vmpp,Impp,Isc,Ncells,a,Print=True)

    # Parameters from Panasonic Photovoltaic module HIT (VBHN330SA16)
    Voc=69.7 #V
    Vmpp=58.0 #V
    Impp=5.70 #A
    Isc=6.07 #A
    Ncells=8*12
    a=1.3
    print ('\nPanasonic:')
    [Rp_HIT,Rs_HIT,Io_HIT,Iph_HIT]=ApproximateModelParameters(Voc/Ncells,Vmpp/Ncells,Impp,Isc,1,a,Rs_guess=0.001,Print=True)