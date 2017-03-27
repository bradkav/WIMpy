import numpy as np
import DMUtils as DMU
import MyParser as MP
from scipy.integrate import quad, cumtrapz
from scipy.interpolate import interp1d, interp2d


class Experiment:
    def __init__(self, exptfile_root, m_min=1.0, m_max=1e3):
        exptfile = "DDexpt/"+exptfile_root
        self.N_iso = int(MP.ReadParam(exptfile, "N_isotopes"))
        self.frac = np.zeros(self.N_iso)
        self.N_p = np.zeros(self.N_iso)
        self.N_n = np.zeros(self.N_iso)
        for i in range(self.N_iso):
            self.frac[i] = float(MP.ReadParam(exptfile, "frac_" + str(i+1)))
            self.N_p[i] = int(MP.ReadParam(exptfile, "N_p_" + str(i+1)))
            self.N_n[i] = int(MP.ReadParam(exptfile, "N_n_" + str(i+1)))
                
        self.m_det = float(MP.ReadParam(exptfile, "m_det"))
        self.E_min = float(MP.ReadParam(exptfile, "E_min"))
        self.E_max = float(MP.ReadParam(exptfile, "E_max"))
        self.exposure = self.m_det*365.0
        self.events = []
        self.R_list = []
        self.Ne_list = []
        self.eventlike = []
        #self.Nevents_tab = [self.TabulateNevents(m_min, m_max, i) for i in range(self.N_iso)]
        #self.rate_tab = [self.TabulateRate(m_min, m_max, i) for i in range(self.N_iso)]
    
    
    def dRdEi(self,E,mx, i):
        R = self.exposure*self.frac[i]*DMU.dRdE(E, self.N_p[i], self.N_n[i], mx, [1.0/self.N_p[i],0.0,0.0,0.0])
        if (R < 1e-30):
            return 1e-30
        else:
            return R
    
    def dRdE(self,E,mx,l):
        R = 0
        for i in range(self.N_iso):
            R += self.frac[i]*DMU.dRdE(E, self.N_p[i], self.N_n[i], mx, l)
        #print R
        if (R < 1e-30):
            return 1e-30
        else:
            return R*self.exposure
    
    def CalcNevents(self,mx,l):
        integ = lambda E: self.dRdE(E,mx,l)
        return quad(integ, self.E_min, self.E_max,epsrel=1e-4)[0]
        
    def CalcNeventsi(self,mx, iso):
        integ = lambda E: self.dRdEi(E,mx, iso)
        return quad(integ, self.E_min, self.E_max,epsrel=1e-4)[0]
        
    def GenerateEvents(self, mx, l):
        N_exp = self.CalcNevents(mx,l)
        No = np.random.poisson(lam=N_exp)
        self.events = np.zeros(No)
        
        #Get the inverse cumulative distribution
        Evals = np.logspace(np.log10(self.E_min), np.log10(self.E_max), 100)
        Rvals = 0.0*Evals
        for i, Ei in enumerate(Evals):
            Rvals[i] = self.dRdE(Ei,mx,l)
        Rcum = cumtrapz(Rvals, Evals, initial=0.0)
        Rcum = Rcum/Rcum[-1]
    
        fx = interp1d(Rcum, Evals)
    
        xvals = np.random.rand(No)
        self.events = fx(xvals)
    
    def TabulateAll(self, mx):
    
        self.Ne_list = np.zeros(self.N_iso)
        for i in range(self.N_iso):
            self.Ne_list[i] = self.CalcNeventsi(mx,i)
            
            
        No = len(self.events)
        self.R_list = np.zeros((No, self.N_iso))
        for i in range(self.N_iso):
            for j in range(No):
                self.R_list[j,i] = self.dRdEi(self.events[j], mx, i)
    
        self.eventlike = np.sum(np.log(self.R_list[:,0]))
    
    def PrintEvents(self):
        print self.events
        
    def TabulateRate(self, m_min, m_max, iso):
        lm_list = np.linspace(np.log10(m_min), np.log10(m_max), 100)
        lE_list = np.linspace(np.log10(self.E_min), np.log10(self.E_max), 100)
        
        lR_grid = np.zeros((100,100))
        for i, lmi in enumerate(lm_list):
            for j, lEj in enumerate(lE_list):
                
                lR_grid[j,i] = np.log10(self.dRdEi(10**lEj, 10**lmi, iso))
                #print 10**lmi, 10**lEj, 10**lR_grid[i,j]
        
        return interp2d(lm_list, lE_list, lR_grid, kind='linear',\
            bounds_error=True)
        
    def TabulateNevents(self, m_min, m_max, iso):
        lm_list = np.linspace(np.log10(m_min), np.log10(m_max), 100)
        lN_list = np.zeros(100)
        
        for i, lmi in enumerate(lm_list):           
            lN_list[i] = np.log10(self.CalcNeventsi(10**lmi,iso))
        
        return interp1d(lm_list, lN_list, kind='linear',\
            bounds_error=True)
        
    def dRdE_tab(self,E,mx,l):
        R = 0
        #print E, mx, 10**self.rate_tab[1](np.log10(mx), np.log10(E))
        #Does this rescaling work??!
        for i in range(self.N_iso):
            R += ((l[0]*self.N_p[i] + l[1]*self.N_n[i])**2.0 + (l[2]*self.N_p[i] + l[3]*self.N_n[i])**2.0)\
                *(10.0**self.rate_tab[i](np.log10(mx), np.log10(E)))
        
        return R
    #    return R*(10.0**self.rate_tab(np.log10(mx), np.log10(E)))

    
    def CalcNevents_tab(self,mx,l):
        R = 0
        #print E, mx, 10**self.rate_tab[1](np.log10(mx), np.log10(E))
        #Does this rescaling work??!
        for i in range(self.N_iso):
            R += ((l[0]*self.N_p[i] + l[1]*self.N_n[i])**2.0 + (l[2]*self.N_p[i] + l[3]*self.N_n[i])**2.0)\
                *(10.0**self.Nevents_tab[i](np.log10(mx)))
        
        return R
    
    