import numpy as np
from numpy import pi


#---Constants-----

#Define the grid of coupling values to evaluate over
Np = 32
Rrange = np.linspace(0.6, 1.05, Np)
frange = np.append(np.linspace(-1.0, -0.975, 3*Np/4), np.linspace(-0.975, -0.94, Np/4))

#Format of coupling vector is always:
#l_p_D, l_n_D, l_p_Dbar, l_n_Dbar                                                                                                                             
#Calculate c_p/c_n                                                                                                                                            
def Calc_cp(lam):
    return np.sqrt((lam[0]**2 + lam[2]**2)/2.0)

def Calc_cn(lam):
    return np.sqrt((lam[1]**2 + lam[3]**2)/2.0)

def Calc_Rpn(lam):
    return Calc_cn(lam)/Calc_cp(lam)

#Calculate f                                                                                                                                                  
def Calc_f(lam):
    return (lam[0]*lam[1] + lam[2]*lam[3])*0.5/(Calc_cp(lam)*Calc_cn(lam))


#Generate point                                                                                             
def GeneratePoint_ind(index):
    vals = np.unravel_index(index-1, (Np,Np), order='C')
    fval = frange[vals[0]]
    Rval = Rrange[vals[1]]
    
    return GeneratePoint(Rval, fval)
                                                  
def GeneratePoint(Rnp, f):
    #Assume that l_p_D = 1 and l_p_Dbar = 0                                                                                                                   
    l_p_D = 1.0
    l_p_Dbar = 0.0

    l_n_D = f*Rnp
    l_n_Dbar = np.sqrt((1.0-f**2))*Rnp

    return np.array([l_p_D, l_n_D, l_p_Dbar, l_n_Dbar])


#---Functions----

def getf(pID):
    vals = np.unravel_index(pID-1, (Np,Np), order='C')
    return frange[vals[0]]
    
def getR_PA(pID):
    vals = np.unravel_index(pID-1, (Np,Np), order='C')
    f = frange[vals[0]]
    return f/np.sqrt(1-f**2)

def getR(pID):
    vals = np.unravel_index(pID-1, (Np,Np), order='C')
    return Rrange[vals[1]]
