from DMUtils import *
from Experiment import *
import sys
import matplotlib.pyplot as pl
from scipy.stats import chi2, norm
import emcee
import CalcParamPoint as CPP
from CalcLikelihood import *


print " "
print "***********************"
print "*   CompareNgrid.py   *"
print "***********************"
print " "

print " Comparing sampling grids for calculating the likelihood..."



#Read parameters from command line
m0 = float(sys.argv[1])
index = int(sys.argv[2])

target_sigma = 1e-46
if (m0 < 30):
    target_sigma = 2e-46
if (m0 > 100):
    target_sigma = 1e-45

#Calculate couplings from index on grid
l = CPP.GeneratePoint_ind(index)

#Rescale couplings to give 'target' DM-proton effective coupling in Xenon
sig0 = (1.973e-14*1.973e-14)*4.0*(reduced_m(1.0, m0))**2.0/np.pi
sig = sig0*0.5*((l[0]*54.0 + l[1]*77.0)**2.0 + (l[2]*54.0 + l[3]*77.0)**2.0)
sig_p = sig/(54+77)**2
l0 = l*np.sqrt(target_sigma/sig_p)


#l0 is in the format [lpD, lnD, lpDb, lnDb]
print " DM mass [GeV]:", m0
print " lambda [GeV^-2]:", l0
print " "

#----Functions----

print " Loading experiments..."
exptlist = ["Xenon2", "Argon", "Silicon"]
N_expt = len(exptlist)
expts = [ Experiment(exptlist[i] + ".txt") for i in range(N_expt)]

print " Generating events..."
for i in range(N_expt):
    expts[i].GenerateEvents(m0, l0)


print " Calculating likelihoods..."
Nmvals = 25
mlist = np.logspace(np.log10(20), np.log10(1000), Nmvals)
likelist = np.zeros((Nmvals, 5))

#Different numbers of grid points to try
Ngrid = [50, 100, 200, 100]
refine = [False, False, False, True]
#In the last case, we use 100 points but refine the grid

likelist_maj = np.zeros((Nmvals, 4))
likelist_dir = np.zeros((Nmvals, 4))

for i, mi in enumerate(mlist):
    print "   ",i+1, "of", Nmvals,": m_x =", mi, "GeV"
    for j in range(4):
        likelist_maj[i,j], likelist_dir[i,j] = CalcLike_grid(mi, expts, Ngrid[j], refine[j])
        
for j in range(4):
    sig = CalcSignificance(np.nanmax(likelist_maj[:,j]), np.nanmax(likelist_dir[:,j]))
    str_extra = ""
    if (refine[j]):
        str_extra = ", refined"
    print " Discrimination significance (N_grid = "+str(Ngrid[j])+str_extra+"):", sig, "sigma"


lines = [":", "-.", "--", "-"]
labels = ["N=50", "N=100", "N=200", "N=100 (refined)"]

pl.figure()
for j in range(4):
    L0 = np.nanmax(likelist_dir[:,j])
    pl.semilogx(mlist, -2*(likelist_maj[:,j]-L0), 'b',\
        linestyle=lines[j], linewidth=1.5)
    pl.semilogx(mlist, -2*(likelist_dir[:,j]-L0), 'r',\
        linestyle=lines[j], linewidth=1.5)

#Add dummy lines for labels
pl.semilogx(1e-30, 1e-30, 'r-',label=r"Dirac", linewidth=1.5)
pl.semilogx(1e-30, 1e-30, 'b-',label=r"Majorana", linewidth=1.5)
for j in range(4):
    pl.semilogx(1e-30, 1e-30, 'k',\
         linestyle=lines[j],label=labels[j], linewidth=1.5)


pl.legend(loc="best", frameon=False)
pl.ylim(-1, 30)
pl.xlim(10, 1000)
pl.axvline(m0, linestyle='--', color='k')
pl.axhline(0, linestyle='--', color='k')
pl.xlabel(r"$m_\chi [GeV]$")
pl.ylabel(r"$-2 \Delta \mathrm{log}\mathcal{L}$")


pl.savefig("plots/GridComparison.pdf")
pl.show()

