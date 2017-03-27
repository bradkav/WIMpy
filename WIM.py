from DMUtils import *
from Experiment import *
import sys
import matplotlib.pyplot as pl
from scipy.stats import chi2, norm
import CalcParamPoint as CPP

ensemble = sys.argv[1]
m0 = float(sys.argv[2])
index = int(sys.argv[3])


if (len(sys.argv) > 4):
    output_folder = sys.argv[4]
else:
    output_folder = "results/"

target_sigma = 1e-46
if (m0 > 90):
    target_sigma = 1e-45

#Recalculate to give the correct numbers!
l = CPP.GeneratePoint_ind(index)
sig0 = (1.973e-14*1.973e-14)*4.0*(reduced_m(1.0, m0))**2.0/np.pi
sig = sig0*0.5*((l[0]*54.0 + l[1]*77.0)**2.0 + (l[2]*54.0 + l[3]*77.0)**2.0)
sig_p = sig/(54+77)**2
l *= np.sqrt(target_sigma/sig_p)

#Should be 312.544 events!

#Sampling paramaters
loglsq_min = -22
loglsq_max = -16

Ngrid = 50


#----Functions----

def CalcLike_grid(mx, Nvals = 100):

    #cp_list = np.logspace(-9, -7, Nvals)**2
    cp_list = np.logspace(loglsq_min, loglsq_max, Nvals)
    #print cp_list
    #cn_list = np.logspace(-9, -7, Nvals)**2
    cn_list = np.logspace(loglsq_min, loglsq_max, Nvals)
    f_list = np.linspace(-1.0,1.0, 2*Nvals)

    for i in range(N_expt):
        expts[i].TabulateAll(mx)

    (CPsq, CNsq, Fr) = np.meshgrid(cp_list, cn_list, f_list)
    
    full_like = 0.0
    for expt in expts:
        No = len(expt.events)
        like = 0.0
        A = np.zeros((expt.N_iso, Nvals, Nvals, 2*Nvals))
    
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
            print " WIM.py: Multi-element experiments not currently supported!"
            return -100
            like += np.sum(np.log(np.dot(A.T,(expt.R_list).T)), axis=3)
        full_like += like
    
    L_maj = np.max(full_like[:,0,:,:])
    L_dir = np.max(full_like)
    
    return L_maj, L_dir

print " Loading experiments for ensemble", ensemble, "..."

if (ensemble == "A"):
    exptlist = ["Xenon2", "Argon", "Silicon"]
elif (ensemble == "B"):
    exptlist = ["Xenon2", "Argon", "Germanium"]
elif (ensemble == "C"):
    exptlist = ["Xenon2", "Argon", "CaWO4"]
elif (ensemble == "D"):
    exptlist = ["Xenon2", "Argon", "Germanium_half","CaWO4_half"]


#exptlist = ["Xenon2", "Argon", "Silicon"]
N_expt = len(exptlist)
expts = [ Experiment(exptlist[i] + ".txt") for i in range(N_expt)]

#print " Generating events..."


print " Calculating likelihoods..."
Nmvals = 100
mlist = np.logspace(np.log10(10), np.log10(1000), Nmvals)

likelist_maj = np.zeros((Nmvals))
likelist_dir = np.zeros((Nmvals))

#Nlist = np.array((4, 25, 50, 100))

#100 is the correct number, but 50 is quick!

Nsamps = 50

sigvals = np.zeros(Nsamps)

for k in range(Nsamps):
    for expt in expts:
        expt.GenerateEvents(m0, l)
    for i, mi in enumerate(mlist):
        #print i+1, mi
        #for j in range(len(Nlist)):
        likelist_maj[i],likelist_dir[i] = CalcLike_grid(mi, Ngrid)
        L0 = np.nanmax(likelist_maj[i])
        L1 = np.nanmax(likelist_dir[i])
        deltaL = L1 - L0
        pval = 1-chi2.cdf(2*deltaL,1)
        #We are using a 2-sided convention for the significance, Z  
        sig = norm.ppf(1-pval/2.0)
        sigvals[k] = sig
    print " Sample", str(k+1), " - Discrimination significance (N_grid = 50):", sig, "sigma"

np.savetxt(output_folder + "Results_p" + str(index)+".txt",sigvals, \
    header="Ensemble "+ ensemble + ", m = " + str(m0) + ", lambda = "+str(l))


print sigvals
print np.median(sigvals)
#pl.figure()
#pl.hist(sigvals, np.linspace(0,6,12))
#pl.show()

