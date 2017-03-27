# coding: utf-8

import numpy as np
from numpy import cos, sin
from scipy.integrate import trapz, cumtrapz, quad
from scipy.interpolate import interp1d
from numpy.random import rand
from scipy.special import sph_jn, erf
import matplotlib as mpl
font = { 'size'   : 14}
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['xtick.minor.size'] = 2
mpl.rcParams['xtick.minor.width'] = 1
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['ytick.minor.size'] = 2
mpl.rcParams['ytick.minor.width'] = 1
mpl.rc('font', **font)

from numpy import pi
import matplotlib.pyplot as pl

from matplotlib.transforms import Affine2D

import mpl_toolkits.axisartist.floating_axes as floating_axes
import  mpl_toolkits.axisartist.angle_helper as angle_helper
from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist.grid_finder import FixedLocator, MaxNLocator, DictFormatter
from scipy.optimize import minimize_scalar, minimize


#-----------------
def loadFormFactors():
    global FFcoeffs
    FFcoeffs = np.loadtxt('data/F19_formfactors.txt')

#---------------------------------------------------------
def calcEta(v,vlag, sigmav,vesc):
    aplus = np.minimum((v+vlag), vesc)/(np.sqrt(2)*sigmav)
    aminus = np.minimum((v-vlag), vesc)/(np.sqrt(2)*sigmav)
    aesc = vesc/(np.sqrt(2)*sigmav)
    #return (np.pi/vlag)*(erf(aplus) - erf(aminus))
    
    vel_integral = 0
    
    N = 1.0/(erf(aesc) - np.sqrt(2.0/np.pi)*(vesc/sigmav)*np.exp(-0.5*(vesc/sigmav)**2))
    
    vel_integral = (0.5/vlag)*(erf(aplus) - erf(aminus))
    vel_integral -= (1.0/(np.sqrt(np.pi)*vlag))*(aplus - aminus)*np.exp(-0.5*(vesc/sigmav)**2)
    
    return 2*np.pi*N*vel_integral
    
#---------------------------------------------------------
def calcMEta(v,vlag,sigmav):
    aplus = (v+vlag)/(np.sqrt(2)*sigmav)
    aminus = (v-vlag)/(np.sqrt(2)*sigmav)
    A = (sigmav**2 + vlag**2 - v**2)*(erf(aplus) - erf(aminus))
    B = np.sqrt(2.0/pi)*sigmav*((vlag-v)*np.exp(-aplus**2) + (vlag + v)*np.exp(-aminus**2))
    return (np.pi/vlag)*(A+B)/(3e5)**2

#---------------------------------------------------------
def calcRT(v, theta, vlag, sigmav):
    prefactor = np.sqrt(2*pi/sigmav)
    return prefactor*np.exp(-((v - vlag*np.cos(theta))/(np.sqrt(2)*sigmav))**2)
    
#---------------------------------------------------------
def calcMRT(v, theta, vlag, sigmav):
    prefactor = np.sqrt(2*pi/sigmav)
    return (1.0/(3e5*3e5))*prefactor*(2*sigmav**2 + (vlag**2)*(np.sin(theta))**2)*np.exp(-((v - vlag*np.cos(theta))/(np.sqrt(2)*sigmav))**2)

#----------------------------------------------------------
def calcSDFormFactor(E, m_N):
    #Fluorine S_00(q)/S_00(0)- arXiv:1301.1457
    
    #Define conversion factor from amu-->keV
    amu = 931.5*1000

    #Convert recoil energy to momentum transfer q in keV
    q1 = np.sqrt(2*m_N*amu*E)
    
    #Convert q into fm^-1
    q2 = q1*(1e-12/1.97e-7)
    b = 1.63 #fm
    u = (q2*b*q2*b)/2.0
    A = 0.10386 - 0.13848*u + 0.06994*u*u - 0.01585*u**3 + 0.00136*u**4
    F = A*np.exp(- 1.0*u)/0.10386
    return F
    
#-----------------------------------------------------------
def calcSIFormFactor(E, m_N, old=False):
    #Helm

    #Define conversion factor from amu-->keV
    amu = 931.5*1e3

    #Convert recoil energy to momentum transfer q in keV
    q1 = np.sqrt(2*m_N*amu*E)

    #Convert q into fm^-1
    q2 = q1*(1e-12/1.97e-7)
    
    #Calculate nuclear parameters
    s = 0.9
    a = 0.52
    c = 1.23*(m_N**(1.0/3.0)) - 0.60
    R1 = np.sqrt(c*c + 7*pi*pi*a*a/3.0 - 5*s*s)
    
    if (old):
        R1 = np.sqrt((1.2**2)*m_N**(2.0/3.0) - 5)
 
    x = q2*R1
    J1 = np.sin(x)/x**2 - np.cos(x)/x
    F = 3*J1/x
    return (F**2)*(np.exp(-(q2*s)**2))

def dRdE(E, N_p, N_n, mx, l): 
  A = N_p + N_n   
  sig = (1.973e-14*1.973e-14)*4.0*(reduced_m(1.0, mx))**2.0/np.pi

  int_factor = 0.5*sig*calcSIFormFactor(E, A)*\
      ((l[0]*N_p + l[1]*N_n)**2.0 + (l[2]*N_p + l[3]*N_n)**2.0)
    
  return rate_prefactor(A, mx)*int_factor*calcEta(vmin(E, A, mx), vlag=232, sigmav=156, vesc=544)


def Nevents(E_min, E_max, N_p, N_n, mx, l):
    integ = lambda x: dRdE(x, N_p, N_n, mx, l)
    return quad(integ, E_min, E_max, epsrel=1e-4)[0]

def rate_prefactor(A, m_x):
    rho0 = 0.3
    mu = 1.78e-27*reduced_m(1.0, m_x)
    return 1.38413e-12*rho0/(4.0*np.pi*m_x*mu*mu)
    
def reduced_m(A, m_x):
    m_A = 0.9315*A
    return (m_A * m_x)/(m_A + m_x)
 
#--------------------------------------------------------
def calcNREFTFormFactor(E,m_A,index,cp,cn):
    #Define conversion factor from amu-->keV
    amu = 931.5*1000

    #Convert recoil energy to momentum transfer q in keV
    q1 = np.sqrt(2*m_A*amu*E)
    #Convert q into fm^-1
    q2 = q1*(1e-12/1.97e-7)
    b = np.sqrt(41.467/(45*m_A**(-1.0/3.0) - 25*m_A**(-2.0/3.0)))
    y = (q2*b/2)**2

    vpp = FFcoeffs[4*(index-1),:]
    vpn = FFcoeffs[4*(index-1)+1,:]
    vnp = FFcoeffs[4*(index-1)+2,:]
    vnn = FFcoeffs[4*(index-1)+3,:]

    Fpp = np.polyval(vpp[::-1],y)
    Fnn = np.polyval(vnn[::-1],y)
    Fnp = np.polyval(vnp[::-1],y)
    Fpn = np.polyval(vpn[::-1],y)
    return (cn*cn*Fnn + cp*cp*Fpp + cn*cp*Fpn + cp*cn*Fnp)*np.exp(-2*y)
    
    
#--------------------------------------------------------
def calcMFormFactor(E, m_A, cp=1, cn=1):
  return calcNREFTFormFactor(E, m_A, 1, cp, cn)

#--------------------------------------------------------
def calcSigma1FormFactor(E, m_A, cp=1, cn=1):
  return calcNREFTFormFactor(E, m_A, 2, cp, cn)  
    
#--------------------------------------------------------
def calcSigma2FormFactor(E, m_A, cp=1, cn=1):
  return calcNREFTFormFactor(E, m_A, 3, cp, cn)   
  
#--------------------------------------------------------
def calcDeltaFormFactor(E, m_A, cp=1, cn=1):
  return calcNREFTFormFactor(E, m_A, 4, cp, cn)

#--------------------------------------------------------
def calcPhi2FormFactor(E, m_A, cp=1, cn=1):
  return calcNREFTFormFactor(E, m_A, 5, cp, cn)

#--------------------------------------------------------
def calcMPhi2FormFactor(E, m_A, cp=1, cn=1):
  return calcNREFTFormFactor(E, m_A, 6, cp, cn)

#--------------------------------------------------------
def calcSigma1DeltaFormFactor(E, m_A, cp=1, cn=1):
  return calcNREFTFormFactor(E, m_A, 7, cp, cn)

#--------------------------------------------------------
def vmin(E, m_N, m_x):
    res = E*0.0
    m_N2 = m_N*0.9315
    mu = (m_N2*m_x)/(m_N2+m_x)
    res =  3e5*np.sqrt((E/1e6)*(m_N2)/(2*mu*mu))
    return res
    
    
    
    
#---------------------------------------------------------
def RateIntegrandSI(E, m_x, theta, vlag=220, sigmav=156):
    m_A = 19 #Fluorine
    return calcRT(vmin(E, m_A, m_x), theta, vlag, sigmav)*calcMFormFactor(E, m_A, 1, 1)

#---------------------------------------------------------
def TotalRateSI(E, m_x, vlag=220, sigmav=156):
    m_A = 19 #Fluorine
    return calcEta(vmin(E, m_A, m_x),  vlag, sigmav)*calcMFormFactor(E, m_A, 1, 1)

#---------------------------------------------------------
def RateIntegrandAnapole(E, m_x, theta, vlag=220, sigmav=156):
    m_A = 19
    m_N = 931.5*1000
    q1 = np.sqrt(2*m_A*m_N*E)
    mu_n = -1.913
    mu_p = 2.793
    A1 = calcMFormFactor(E, m_A, 1, 0)
    B1 = calcDeltaFormFactor(E, m_A, 1, 0)
    B2 = -mu_p*calcSigma1DeltaFormFactor(E, m_A, 1, 0)
    B3 = -0.5*mu_n*(calcSigma1DeltaFormFactor(E, m_A, 1, 1) - calcSigma1DeltaFormFactor(E, m_A, 1, 0) - calcSigma1DeltaFormFactor(E, m_A, 0, 1))
    B4 = 0.25*(calcSigma1FormFactor(E, m_A, mu_p, mu_n))
    
    #A1 = 0
    #A1 = 0
    #B1 = 0
    #B2 = 0
    #B3 = 0
    #B4 = 0
    
    #NOTE: Proportionality is not exact...
    pref = 1.0
    #pref = (q1**-4)
    #This is not finished...
    #print q1/m_N, 81*sigmav/3e5
    return pref*(calcMRT(vmin(E, m_A, m_x), theta, vlag, sigmav)*A1 + calcRT(vmin(E, m_A, m_x), theta, vlag, sigmav)*((q1/m_N)**2)*(B1+B2+B3+B4))
    #return pref*(calcMRT(vmin(E, m_A, m_x), theta, vlag, sigmav) + calcRT(vmin(E, m_A, m_x), theta, vlag, sigmav)*((q1/m_N)**2))*A1

#---------------------------------------------------------
def TotalRateAnapole(E, m_x, vlag=220, sigmav=156):
    m_A = 19 #Fluorine
    m_N = 931.5*1000
    q1 = np.sqrt(2*m_A*m_N*E)
    mu_n = -1.913
    mu_p = 2.793
    A1 = calcMFormFactor(E, m_A, 1, 0)
    B1 = calcDeltaFormFactor(E, m_A, 1, 0)
    B2 = -mu_p*calcSigma1DeltaFormFactor(E, m_A, 1, 0)
    B3 = -0.5*mu_n*(calcSigma1DeltaFormFactor(E, m_A, 1, 1) - calcSigma1DeltaFormFactor(E, m_A, 1, 0) - calcSigma1DeltaFormFactor(E, m_A, 0, 1))
    B4 = 0.25*(calcSigma1FormFactor(E, m_A, mu_p, mu_n))

    #NOTE: Proportionality is not exact...
    #B1 = 0
    #B2 = 0
    #B3 = 0
    #B4 = 0
    #A1 = 0

    pref = 1.0
    #pref = (q1**-4)

    #This is not finished...
    return pref*(calcMEta(vmin(E, m_A, m_x), vlag, sigmav)*A1 + calcEta(vmin(E, m_A, m_x), vlag, sigmav)*((q1/m_N)**2)*(B1+B2+B3+B4))
  
  
   
   
   
   
   
   
#---------------------------------------------------------
def RateIntegrandSD(E, m_x, theta, vlag=220, sigmav=156):
    m_N = 19 #Fluorine
    return calcRT(vmin(E, m_N, m_x), theta, vlag, sigmav)*(calcSigma1FormFactor(E, m_N) + calcSigma2FormFactor(E, m_N))

#---------------------------------------------------------
def TotalRateSD(E, m_x, vlag=220, sigmav=156):
    m_N = 19 #Fluorine
    return calcEta(vmin(E, m_N, m_x), vlag, sigmav)*(calcSigma1FormFactor(E, m_N) + calcSigma2FormFactor(E, m_N))
 
#---------------------------------------------------------
#def RateIntegrandSD(E, m_x, theta, vlag=220, sigmav=156):
#    m_N = 19 #Fluorine
#    return calcRT(vmin(E, m_N, m_x), theta, vlag, sigmav)*calcSDFormFactor(E, m_N)
    
#---------------------------------------------------------
def RateIntegrandO3(E, m_x, theta, vlag=220, sigmav=156):
    m_N = 19 #Fluorine
    amu = 931.5*1000
    q1 = np.sqrt(2*m_N*amu*E)
    A = 0.5*calcMRT(vmin(E, m_N, m_x), theta, vlag, sigmav)*calcSigma1FormFactor(E, m_N)
    B = ((q1/amu)**2)*calcRT(vmin(E, m_N, m_x), theta, vlag, sigmav)*calcPhi2FormFactor(E, m_N)
    return (q1/amu)**2*(A+B)    
    
#---------------------------------------------------------
def TotalRateO3(E, m_x, vlag=220, sigmav=156):
    m_N = 19 #Fluorine
    amu = 931.5*1000
    q1 = np.sqrt(2*m_N*amu*E)
    A = 0.5*calcMEta(vmin(E, m_N, m_x), vlag, sigmav)*calcSigma1FormFactor(E, m_N)
    B = ((q1/amu)**2)*calcEta(vmin(E, m_N, m_x), vlag, sigmav)*calcPhi2FormFactor(E, m_N)
    return (q1/amu)**2*(A+B)    
    
#---------------------------------------------------------
def RateIntegrandO5(E, m_x, theta, vlag=220, sigmav=156):
    m_N = 19 #Fluorine
    amu = 931.5*1000
    q1 = np.sqrt(2*m_N*amu*E)
    A = calcMRT(vmin(E, m_N, m_x), theta, vlag, sigmav)*calcMFormFactor(E, m_N)
    B = ((q1/amu)**2)*calcRT(vmin(E, m_N, m_x), theta, vlag, sigmav)*calcDeltaFormFactor(E, m_N)
    return (q1/amu)**2*(A+B)    

#---------------------------------------------------------
def TotalRateO5(E, m_x, vlag=220, sigmav=156):
    m_N = 19 #Fluorine
    amu = 931.5*1000
    q1 = np.sqrt(2*m_N*amu*E)
    A = calcMEta(vmin(E, m_N, m_x), vlag, sigmav)*calcMFormFactor(E, m_N)
    B = ((q1/amu)**2)*calcEta(vmin(E, m_N, m_x), vlag, sigmav)*calcDeltaFormFactor(E, m_N)
    return (q1/amu)**2*(A+B)  

#---------------------------------------------------------
def RateIntegrandO6(E, m_x, theta, vlag=220, sigmav=156):
    m_N = 19 #Fluorine
    amu = 931.5*1000
    q1 = np.sqrt(2*m_N*amu*E)
    A = ((q1/amu)**4)*calcRT(vmin(E, m_N, m_x), theta, vlag, sigmav)*calcSigma2FormFactor(E, m_N)
    return A
    
#---------------------------------------------------------
def TotalRateO6(E, m_x, vlag=220, sigmav=156):
    m_N = 19 #Fluorine
    amu = 931.5*1000
    q1 = np.sqrt(2*m_N*amu*E)
    A = ((q1/amu)**4)*calcEta(vmin(E, m_N, m_x), vlag, sigmav)*calcSigma2FormFactor(E, m_N)
    return A
    

#---------------------------------------------------------
def RateIntegrandO7(E, m_x, theta, vlag=220, sigmav=156):
    m_N = 19 #Fluorine
    return calcMRT(vmin(E, m_N, m_x), theta, vlag, sigmav)*calcSigma1FormFactor(E, m_N)

#---------------------------------------------------------
def TotalRateO7(E, m_x, vlag=220, sigmav=156):
    m_N = 19 #Fluorine
    return calcMEta(vmin(E, m_N, m_x),  vlag, sigmav)*calcSigma1FormFactor(E, m_N)


#---------------------------------------------------------
def RateIntegrandO8(E, m_x, theta, vlag=220, sigmav=156):
    m_N = 19 #Fluorine
    amu = 931.5*1000
    q1 = np.sqrt(2*m_N*amu*E)
    A = calcMRT(vmin(E, m_N, m_x), theta, vlag, sigmav)*calcMFormFactor(E, m_N)
    B = ((q1/amu)**2)*calcRT(vmin(E, m_N, m_x), theta, vlag, sigmav)*calcDeltaFormFactor(E, m_N)
    return A+B
    
#---------------------------------------------------------
def TotalRateO8(E, m_x, vlag=220, sigmav=156):
    m_N = 19 #Fluorine
    amu = 931.5*1000
    q1 = np.sqrt(2*m_N*amu*E)
    A = calcMEta(vmin(E, m_N, m_x), vlag, sigmav)*calcMFormFactor(E, m_N)
    B = ((q1/amu)**2)*calcEta(vmin(E, m_N, m_x), vlag, sigmav)*calcDeltaFormFactor(E, m_N)
    return A+B
  
#---------------------------------------------------------
def RateIntegrandO9(E, m_x, theta, vlag=220, sigmav=156):
    m_N = 19 #Fluorine
    amu = 931.5*1000
    q1 = np.sqrt(2*m_N*amu*E)
    A = ((q1/amu)**2)*calcRT(vmin(E, m_N, m_x), theta, vlag, sigmav)*calcSigma1FormFactor(E, m_N)
    return A

#---------------------------------------------------------
def TotalRateO9(E, m_x, vlag=220, sigmav=156):
    m_N = 19 #Fluorine
    amu = 931.5*1000
    q1 = np.sqrt(2*m_N*amu*E)
    A = ((q1/amu)**2)*calcEta(vmin(E, m_N, m_x), vlag, sigmav)*calcSigma1FormFactor(E, m_N)
    return A
    
#---------------------------------------------------------
def RateIntegrandO8_9(E, m_x, theta, vlag=220, sigmav=156):
    m_N = 19 #Fluorine
    amu = 931.5*1000
    q1 = np.sqrt(2*m_N*amu*E)
    A = ((q1/amu)**2)*calcRT(vmin(E, m_N, m_x), theta, vlag, sigmav)*calcSigma1FormFactor(E, m_N)
    return A
    
 
#---------------------------------------------------------
def RateIntegrandO10(E, m_x, theta, vlag=220, sigmav=156):
    m_N = 19 #Fluorine
    amu = 931.5*1000
    q1 = np.sqrt(2*m_N*amu*E)
    A = ((q1/amu)**2)*calcRT(vmin(E, m_N, m_x), theta, vlag, sigmav)*calcSigma2FormFactor(E, m_N)
    return A
 
#---------------------------------------------------------
def TotalRateO10(E, m_x, vlag=220, sigmav=156):
    m_N = 19 #Fluorine
    amu = 931.5*1000
    q1 = np.sqrt(2*m_N*amu*E)
    A = ((q1/amu)**2)*calcEta(vmin(E, m_N, m_x), vlag, sigmav)*calcSigma2FormFactor(E, m_N)
    return A
  
#---------------------------------------------------------
def RateIntegrandO11(E, m_x, theta, vlag=220, sigmav=156):
    m_N = 19 #Fluorine
    amu = 931.5*1000
    q1 = np.sqrt(2*m_N*amu*E)
    A = ((q1/amu)**2)*calcRT(vmin(E, m_N, m_x), theta, vlag, sigmav)*calcMFormFactor(E, m_N)
    return A

#---------------------------------------------------------
def TotalRateO11(E, m_x,  vlag=220, sigmav=156):
    m_N = 19 #Fluorine
    amu = 931.5*1000
    q1 = np.sqrt(2*m_N*amu*E)
    A = ((q1/amu)**2)*calcEta(vmin(E, m_N, m_x), vlag, sigmav)*calcMFormFactor(E, m_N)
    return A
  
#---------------------------------------------------------
def RateIntegrandO12(E, m_x, theta, vlag=220, sigmav=156):
    m_N = 19 #Fluorine
    amu = 931.5*1000
    q1 = np.sqrt(2*m_N*amu*E)
    A = calcMRT(vmin(E, m_N, m_x), theta, vlag, sigmav)*(calcSigma2FormFactor(E, m_N) + 0.5*calcSigma1FormFactor(E, m_N))
    B = ((q1/amu)**2)*calcRT(vmin(E, m_N, m_x), theta, vlag, sigmav)*calcPhi2FormFactor(E, m_N)
    return A+B

#---------------------------------------------------------
def TotalRateO12(E, m_x, vlag=220, sigmav=156):
    m_N = 19 #Fluorine
    amu = 931.5*1000
    q1 = np.sqrt(2*m_N*amu*E)
    A = calcMEta(vmin(E, m_N, m_x), vlag, sigmav)*(calcSigma2FormFactor(E, m_N) + 0.5*calcSigma1FormFactor(E, m_N))
    B = ((q1/amu)**2)*calcEta(vmin(E, m_N, m_x), vlag, sigmav)*calcPhi2FormFactor(E, m_N)
    return A+B

#---------------------------------------------------------
def RateIntegrandO13(E, m_x, theta, vlag=220, sigmav=156):
    m_N = 19 #Fluorine
    amu = 931.5*1000
    q1 = np.sqrt(2*m_N*amu*E)
    A = calcMRT(vmin(E, m_N, m_x), theta, vlag, sigmav)*(calcSigma2FormFactor(E, m_N))
    return ((q1/amu)**2)*A
 
#---------------------------------------------------------
def TotalRateO13(E, m_x, vlag=220, sigmav=156):
    m_N = 19 #Fluorine
    amu = 931.5*1000
    q1 = np.sqrt(2*m_N*amu*E)
    A = calcMEta(vmin(E, m_N, m_x), vlag, sigmav)*(calcSigma2FormFactor(E, m_N))
    return ((q1/amu)**2)*A
  
  
#---------------------------------------------------------
def RateIntegrandO14(E, m_x, theta, vlag=220, sigmav=156):
    m_N = 19 #Fluorine
    amu = 931.5*1000
    q1 = np.sqrt(2*m_N*amu*E)
    A = calcMRT(vmin(E, m_N, m_x), theta, vlag, sigmav)*(calcSigma1FormFactor(E, m_N))
    return ((q1/amu)**2)*A

#---------------------------------------------------------
def TotalRateO14(E, m_x, vlag=220, sigmav=156):
    m_N = 19 #Fluorine
    amu = 931.5*1000
    q1 = np.sqrt(2*m_N*amu*E)
    A = calcMEta(vmin(E, m_N, m_x), vlag, sigmav)*(calcSigma1FormFactor(E, m_N))
    return ((q1/amu)**2)*A
    
#---------------------------------------------------------
def RateIntegrandO15(E, m_x, theta, vlag=220, sigmav=156):
    m_N = 19 #Fluorine
    amu = 931.5*1000
    q1 = np.sqrt(2*m_N*amu*E)
    A = 0.5*calcMRT(vmin(E, m_N, m_x), theta, vlag, sigmav)*(calcSigma1FormFactor(E, m_N))
    B = ((q1/amu)**2)*calcRT(vmin(E, m_N, m_x), theta, vlag, sigmav)*calcPhi2FormFactor(E, m_N)
    return ((q1/amu)**4)*(A+B)
    
#---------------------------------------------------------
def TotalRateO15(E, m_x, vlag=220, sigmav=156):
    m_N = 19 #Fluorine
    amu = 931.5*1000
    q1 = np.sqrt(2*m_N*amu*E)
    A = 0.5*calcMEta(vmin(E, m_N, m_x), vlag, sigmav)*(calcSigma1FormFactor(E, m_N))
    B = ((q1/amu)**2)*calcEta(vmin(E, m_N, m_x), vlag, sigmav)*calcPhi2FormFactor(E, m_N)
    return ((q1/amu)**4)*(A+B)

#---------------------------------------------------------
def RateIntegrandLR1(E, m_x, theta, vlag=220, sigmav=156):
    m_N = 19 #Fluorine
    amu = 931.5*1000
    q1 = np.sqrt(2*m_N*amu*E)
    A = calcRT(vmin(E, m_N, m_x), theta, vlag, sigmav)*calcMFormFactor(E, m_N)
    return A/((q1/amu)**4)
    
#---------------------------------------------------------
def RateIntegrandTest(E, m_x, theta, vlag=220, sigmav=156):
    m_N = 19 #Fluorine
    amu = 931.5*1000
    q1 = np.sqrt(2*m_N*amu*E)
    m = amu*m_N
    return calcMRT(vmin(E, m_N, m_x), theta, vlag, sigmav)*calcSigma2FormFactor(E, m_N)

#---------------------------------------------------------
def Rate(m_x, E_min, theta, vlag=220, sigmav=156, oper='SI'):
    E_max = 50.00
    if (oper == 'SI'):
        return quad(RateIntegrandSI, E_min, E_max, args=(m_x, theta, vlag, sigmav))[0]
    if (oper == 'SD'):
        return quad(RateIntegrandSD, E_min, E_max, args=(m_x, theta, vlag, sigmav))[0]
    if (oper == 'O3'):
        return quad(RateIntegrandO3, E_min, E_max, args=(m_x, theta, vlag, sigmav))[0]
    if (oper == 'O5'):
        return quad(RateIntegrandO5, E_min, E_max, args=(m_x, theta, vlag, sigmav))[0]
    if (oper == 'O6'):
        return quad(RateIntegrandO6, E_min, E_max, args=(m_x, theta, vlag, sigmav))[0]
    if (oper == 'O7'):
        return quad(RateIntegrandO7, E_min, E_max, args=(m_x, theta, vlag, sigmav))[0]
    if (oper == 'O8'):
        return quad(RateIntegrandO8, E_min, E_max, args=(m_x, theta, vlag, sigmav))[0]
    if (oper == 'O9'):
        return quad(RateIntegrandO9, E_min, E_max, args=(m_x, theta, vlag, sigmav))[0]
    if (oper == 'O10'):
        return quad(RateIntegrandO10, E_min, E_max, args=(m_x, theta, vlag, sigmav))[0]
    if (oper == 'O11'):
        return quad(RateIntegrandO11, E_min, E_max, args=(m_x, theta, vlag, sigmav))[0]
    if (oper == 'O12'):
        return quad(RateIntegrandO12, E_min, E_max, args=(m_x, theta, vlag, sigmav))[0]
    if (oper == 'O13'):
        return quad(RateIntegrandO13, E_min, E_max, args=(m_x, theta, vlag, sigmav))[0]
    if (oper == 'O14'):
        return quad(RateIntegrandO14, E_min, E_max, args=(m_x, theta, vlag, sigmav))[0]
    if (oper == 'O15'):
        return quad(RateIntegrandO15, E_min, E_max, args=(m_x, theta, vlag, sigmav))[0]
    if (oper == 'LR1'):
        return quad(RateIntegrandLR1, E_min, E_max, args=(m_x, theta, vlag, sigmav))[0]
    if (oper == 'Test'):
        return quad(RateIntegrandTest, E_min, E_max, args=(m_x, theta, vlag, sigmav))[0]
    if (oper == 'Anapole'):
        return quad(RateIntegrandAnapole, E_min, E_max, args=(m_x, theta, vlag, sigmav))[0]

#---------------------------------------------------------
def TotalRate(E, m_x, vlag=220, sigmav=156, oper='SI'):

    if ((oper == 'O1') or (oper == 'SI')):
        return TotalRateSI(E, m_x, vlag, sigmav)
    if (oper == 'O3'):
        return TotalRateO3(E, m_x, vlag, sigmav)
    if ((oper == 'O4') or (oper == 'SD')):
        return TotalRateSD(E, m_x, vlag, sigmav)
    if (oper == 'O5'):
        return TotalRateO5(E, m_x, vlag, sigmav)
    if (oper == 'O6'):
        return TotalRateO6(E, m_x, vlag, sigmav)
    if (oper == 'O7'):
        return TotalRateO7(E, m_x, vlag, sigmav)
    if (oper == 'O8'):
        return TotalRateO8(E, m_x, vlag, sigmav)
    if (oper == 'O9'):
        return TotalRateO9(E, m_x, vlag, sigmav)
    if (oper == 'O10'):
        return TotalRateO10(E, m_x, vlag, sigmav)
    if (oper == 'O11'):
        return TotalRateO11(E, m_x, vlag, sigmav)
    if (oper == 'O12'):
        return TotalRateO12(E, m_x, vlag, sigmav)
    if (oper == 'O13'):
        return TotalRateO13(E, m_x, vlag, sigmav)
    if (oper == 'O14'):
        return TotalRateO14(E, m_x, vlag, sigmav)
    if (oper == 'O15'):
        return TotalRateO15(E, m_x, vlag, sigmav)
    if (oper == 'Anapole'):
        return TotalRateAnapole(E, m_x, vlag, sigmav)

#---------------------------------------------------------
def FindMaximum(m_x, E_min, vlag=220, sigmav=156, oper='SI'):
    f = lambda t, m, E, vl, sig, op: - Rate(m, E, t, vl, sig, op) 
    return minimize_scalar(f, bounds=(0, pi), args=(m_x, E_min, vlag, sigmav, oper), method='bounded')
    
    
#---------------------------------------------------------
def arcLengths(a, xi, yi, zi):
    theta = a[0]
    phi = a[1]
    #print "theta in arcLengths is: ", theta
    #print "phi in arcLengths is: ", phi
    xa = np.sin(theta)*np.cos(phi)
    ya = np.sin(theta)*np.sin(phi)
    za = np.cos(theta)
    M = np.sum(np.arccos((xa*xi + ya*yi + za*zi)))
    return M
    
    
#---------------------------------------------------------
def calcModifiedRW(xi,yi,zi):
    N = xi.size
    #Calculate the Rayleigh statistic
    R = np.sqrt(np.sum(xi)**2 + np.sum(yi)**2 + np.sum(zi)**2)
    #Calculate RW statistic
    W = 3.0*R**2/(1.0*N)
    MRW = (1.0 - 1.0/(2.0*N))*W + W**2/(10.0*N)
    return MRW

    
#---------------------------------------------------------
def FindMedianDir(xi,yi,zi):
    xbar = np.mean(xi)
    ybar = np.mean(yi)
    zbar = np.mean(zi)
    tbar = np.arccos(zbar)
    pbar = np.mean(np.arctan(xi/yi))
    tlim = (tbar - 0.5*pi, tbar+0.5*pi)
    #return minimize(arcLengths, [tbar,0],args=(xi,yi,zi), method='COBYLA', bounds=(tlim, (0,2*pi)))
    return minimize(arcLengths, [tbar,0],args=(xi,yi,zi), method='COBYLA')
    
#---------------------------------------------------------
def circConstrain(theta):
    res = theta*1.0
    res[theta > 180.0] = theta[theta > 180.0] - 360.0
    res[theta < -180.0] = theta[theta < -180.0] + 360.0
    return res
    
#---------------------------------------------------------
def calcPercentile(y, p):
    #res = p*0.0
    edges = np.linspace(np.min(y), np.max(y), 101)
    x = edges[0:-1] + (edges[1] - edges[0])/2.0
    n = np.histogram(y, edges)[0]
    cum = 1.0*np.cumsum(n)/y.size
    #pl.figure()
    #pl.plot(x,cum)
    #pl.show()
    inv = interp1d(cum,x,bounds_error=False, fill_value=0.0)
    return inv(p)

#---------------------------------------------------------
def MC_generateDir(interpf, N_e):
    ui = rand(N_e)
    ti = interpf(ui)*pi/180.0
    phii = 2*pi*rand(N_e)
    
    xi = np.cos(phii)*np.sin(ti)
    yi = np.sin(phii)*np.sin(ti)
    zi = np.cos(ti)
    
    return (xi,yi,zi)
    
    
#---------------------------------------------------------
def MC_generate2d(E,interpf, N_e, norm):
    i = 0
    xi = np.zeros(N_e)
    yi = np.zeros(N_e)
    zi = np.zeros(N_e)
    Ei = np.zeros(N_e)
    
    while (i < N_e):
        Ei[i] = (np.max(E) - np.min(E))*rand(1) + np.min(E)
        ti = np.arccos(2.0*rand(1) - 1)
        phii = 2.0*pi*rand(1)
        p = rand(1)
        if (p < interpf(Ei[i], 100, ti, vlag=220, sigmav=156)/norm):
            xi[i] = np.cos(phii)*np.sin(ti)
            yi[i] = np.sin(phii)*np.sin(ti)
            zi[i] = np.cos(ti) 
            i = i+1  
    
    return (Ei,xi,yi,zi)
    

