from DMUtils import *
from Experiment import *
import sys
import matplotlib.pyplot as pl
from scipy.stats import chi2, norm
import emcee
import CalcParamPoint as CPP

#---Constants-----
Np = 32
Rrange = np.linspace(0.6, 1.1, Np)
frange = np.append(np.linspace(-1.0, -0.975, 3*Np/4), np.linspace(-0.975,-0.95, Np/4))


m0 = 50
#l0 = np.array([1.54074e-8, -9.95741e-9, 0, 8.06107e-10])

#p = 107
#l0 = np.array([1.39016e-08, -1.05486e-08, 0, 8.53969e-10])

#p = 800
l0 = np.array([2.65741e-09, -2.85008e-09, 0, 6.49539e-10])

print " l0 = ", l0

print " l_n/l_p =",CPP.Calc_cn(l0)/CPP.Calc_cp(l0)
print " f =", CPP.Calc_f(l0)

#l0 = np.array([1.4074e-8, -1.95741e-11, -1e-8, 8.06107e-10])

#Should be 312.544 events!


#Double check these values!
#c0 = -12
#c1 = -7
c0 = -22
c1 = -16


ndim = 3
nwalkers = 10
nsteps = 100

Tchain = 1.0

print " Loading experiments..."
exptlist = ["Xenon2", "Argon", "Silicon"]
N_expt = len(exptlist)
expts = [ Experiment(exptlist[i] + ".txt") for i in range(N_expt)]

#expt1 = Experiment("Xenon.txt")
#print expt1.CalcNevents(m0,l0)
#expt1.GenerateEvents(m0, l0)
#expt1.PrintEvents()

print " Generating events..."
for i in range(N_expt):
    expts[i].GenerateEvents(m0, l0)
    #expts[i].TabulateRate(1e1, 1e3)


def lnprior(x):
    (lcp, lcn, f) = x
    if ((-11 < lcp < -5)and(-11 < lcn < -5)and(-1 < f < 1)):
        return 0.0
    return -np.inf

def CalcExptLike(expt, mx, l, T=1.0):

    #Poisson likelihood
    No = len(expt.events)
    Ne = expt.CalcNevents_tab(mx,l)
    #print Ne
    PL = -Ne + No*np.log(Ne) 

    #print expt, mx, l
    #print PL
    #Event-by-event likelihood
    for i in range(No):
        PL += np.log(expt.exposure*expt.dRdE(expt.events[i], mx, l)/Ne)

    if np.isnan(PL):
        return -1e30
    else:
        return PL*1.0/T

#Get a (sorted) chain with two columns - values and log-likelihoods
def ProcessChain(sampler):
    chain = np.column_stack((sampler.flatchain[:],sampler.flatlnprobability))
    chain = chain[chain[:,0].argsort()]
    return chain
        
def CalcExptLike_grid(mx, Nvals = 100):

    #cp_list = np.logspace(-9, -7, Nvals)**2
    cp_list = np.logspace(c0, c1, Nvals)
    #print cp_list
    #cn_list = np.logspace(-9, -7, Nvals)**2
    cn_list = np.logspace(c0, c1, Nvals)
    f_list = np.linspace(-1.0,1.0, 4*Nvals)

    for i in range(N_expt):
        expts[i].TabulateAll(mx)

    (CPsq, CNsq, Fr) = np.meshgrid(cp_list, cn_list, f_list)
    
    full_like = 0.0
    for expt in expts:
        No = len(expt.events)
        like = 0.0
        A = np.zeros((expt.N_iso, Nvals, Nvals, 4*Nvals))
    
        for i in range(expt.N_iso):
            A[i, :, :, :] = 2.0*(CPsq*(expt.N_p[i] + Fr*expt.N_n[i])**2 + CNsq*expt.N_n[i]**2)
        
        #print A*expt.Ne_list
    
        #This definitely doesn't work!
        #like = -np.dot(A.T, expt.Ne_list)
        

        like = -A*expt.Ne_list
        #print "   ",No, np.min(A*expt.Ne_list),np.max(A*expt.Ne_list)
        #print -like
        #print like
        #print "3"
        if (expt.N_iso == 1):
            like += expt.eventlike + No*np.log(A[0,:,:,:])
        else:
            like += np.sum(np.log(np.dot(A.T,(expt.R_list).T)), axis=3)
        #print "4"
        full_like += like
    
    ind = np.argmax(full_like[:,0,:,:])
    ind2 = np.argmax(full_like)

    #print full_like.shape

    print "    delta-chi-sq:", -2*(np.max(full_like[:,:,:,:]) - np.max(full_like[:,0,:,:]))
    print "    Best-fit (Maj.):",np.log10(np.sqrt(CPsq.flatten()[ind])), np.sqrt(CPsq.flatten()[ind])*Fr.flatten()[ind]
    print "    Best-fit (Dir.):",np.sqrt(CPsq.flatten()[ind2]), np.sqrt(CPsq.flatten()[ind2])*Fr.flatten()[ind],np.sqrt(CNsq.flatten()[ind2])
    print " "

    if (mx > 50):
        pl.figure()
        pl.contourf(np.log10(np.sqrt(cp_list)), f_list, full_like[0,0,:,:].T - np.max(full_like[0,0,:,:]),np.linspace(-20,1,101))
        pl.plot(np.log10(np.sqrt(CPsq.flatten()[ind])), Fr.flatten()[ind], 'gs')
        pl.title(r'$N_\mathrm{grid} = '+ str(Nvals)+'$')
        pl.colorbar()
        pl.show()
    
    #Use "MAP"?
    
    #Monte carlo each value of the mass...
    
    #print np.argmax(full_like[:,0,:,:])

    #print cn_list[0]
    #ind = np.argmax(full_like)
    #print (A.flatten())[ind]*expts[0].Ne_list
    #print ind
    #print CPsq.flatten()[ind], CNsq.flatten()[ind], Fr.flatten()[ind]
    L_maj = np.max(full_like[:,0,:,:])
    L_dir = np.max(full_like)
    
    return L_maj, L_dir


def lnprob(x):
    lp = lnprior(x)
    if not np.isfinite(lp):
        return -np.inf

    cp = 10**x[0]
    cn = 10**x[1]
    f = x[2]
    
    like_full = 0.0
    for expt in expts:
        like = 0.0
        A = np.zeros(expt.N_iso)
        for i in range(expt.N_iso):
            
            #A[i] = 2.0*(cp**2*expt.N_p[i]**2 + cn**2*expt.N_n[i]**2 + 2.0*cp*cn*f*expt.N_p[i]*expt.N_n[i])
            A[i] = 2.0*(cp**2*(expt.N_p[i] + f*expt.N_n[i])**2 + cn**2*expt.N_n[i]**2)
        like = -np.dot(A.T, expt.Ne_list)
        if (expt.N_iso == 1):
            like += expt.eventlike + np.log(A[0])*len(expt.events)
        else:
            like += np.sum(np.log(np.dot(A.T,(expt.R_list).T)))
        if (np.isnan(like)):
            return -np.inf
        like_full += like
    return lp + like_full/Tchain

def CalcExptLike_MC(mx):

    for i in range(N_expt):
        expts[i].TabulateAll(mx)

    
    pos = [(-6.1,-6.2, 0.1) + 0.25*np.random.randn(ndim) for i in range(nwalkers)]
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=())
    sampler.run_mcmc(pos, nsteps)
    #chain = ProcessChain(sampler)
    #print chain.shape
    #print np.max(sampler.flatlnprobability)
    return np.max(sampler.flatlnprobability)*Tchain
    

    
#E_list = np.linspace(expts[0].E_min, expts[0].E_max, 100)
#R_list = 0.0*E_list
#for i, Ei in enumerate(E_list):
#    R_list[i] = expts[0].dRdE_tab(Ei, m0, l0)

#pl.figure()
#pl.loglog(E_list, R_list)
#pl.ylim(1e-10,100)
#pl.show()

print " Fix normalisation of the couplings etc..."

print " Calculating likelihoods..."
Nmvals = 200
mlist = np.logspace(np.log10(200.1), np.log10(1000), Nmvals)
likelist = np.zeros((Nmvals, 5))

likelist_low = np.zeros((Nmvals))
likelist_high = np.zeros((Nmvals))
likelist_max = np.zeros((Nmvals))

likelist_maj = np.zeros((Nmvals))
likelist_dir = np.zeros((Nmvals))

likelist_maj2 = np.zeros((Nmvals))
likelist_dir2 = np.zeros((Nmvals))

likelist_maj3 = np.zeros((Nmvals))
likelist_dir3 = np.zeros((Nmvals))

#Nlist = np.array((4, 25, 50, 100))

#100 is the correct number, but 50 is quick!

N0 = 200
N1 = 100
N2 = 50

for i, mi in enumerate(mlist):
    print i+1, mi
    #for j in range(len(Nlist)):
    likelist_maj3[i],likelist_dir3[i] = CalcExptLike_grid(mi, N2)
    likelist_maj2[i],likelist_dir2[i] = CalcExptLike_grid(mi, N1)
    likelist_maj[i],likelist_dir[i] = CalcExptLike_grid(mi, N0)


    
L0 = np.nanmax(likelist_maj)
L1 = np.nanmax(likelist_dir)
deltaL = L1 - L0
pval = 1-chi2.cdf(2*deltaL,1)
sig = norm.ppf(1-pval/2.0)

L0_2 = np.nanmax(likelist_maj2)
L1_2 = np.nanmax(likelist_dir2)
deltaL = L1_2 - L0_2
pval = 1-chi2.cdf(2*deltaL,1)
sig2 = norm.ppf(1-pval/2.0)

L0_3 = np.nanmax(likelist_maj3)
L1_3 = np.nanmax(likelist_dir3)
deltaL = L1_3 - L0_3
pval = 1-chi2.cdf(2*deltaL,1)
sig3 = norm.ppf(1-pval/2.0)


#We are using a 2-sided convention for the significance, Z                         

print " Discrimination significance (N_grid = "+str(N2)+"):", sig3, "sigma"
print " Discrimination significance (N_grid = "+str(N1)+"):", sig2, "sigma"
print " Discrimination significance (N_grid = "+str(N0)+"):", sig, "sigma"

 
"""
pl.figure()
for j in range(len(Nlist)):
    pl.semilogx(mlist, -2*(likelist[:,j]-L0), label=r"$N_\mathrm{grid} = "+str(Nlist[j])+"$")

pl.legend(loc="best", frameon=False)
pl.ylim(-1, 100)
pl.axvline(50, linestyle='--', color='k')
pl.axhline(0, linestyle='--', color='k')
pl.show()
"""

pl.figure()
pl.semilogx(mlist, -2*(likelist_maj-L1), 'b-',label=r"Majorana", linewidth=1.5)
pl.semilogx(mlist, -2*(likelist_dir-L1), 'g-',label=r"Dirac", linewidth=1.5)
pl.semilogx(mlist, -2*(likelist_maj2-L1),  'b--',linewidth=1.5)
pl.semilogx(mlist, -2*(likelist_dir2-L1), 'g--',linewidth=1.5)
pl.semilogx(mlist, -2*(likelist_maj3-L1),  'b:',linewidth=1.5)
pl.semilogx(mlist, -2*(likelist_dir3-L1), 'g:',linewidth=1.5)

pl.legend(loc="best", frameon=False)
pl.ylim(-1, 100)
pl.axvline(50, linestyle='--', color='k')
pl.axhline(0, linestyle='--', color='k')
pl.show()

#print likelist
#print likelist
#pl.figure()
#pl.semilogx(mlist, -2*(likelist - np.nanmax(likelist)))

#pl.axvline(50, linestyle='--', color='k')
#pl.show()

#pl.figure()
#pl.hist(expt1.events,100)
#pl.show()

