'''
Module to compute the autocorrelation function and autocorrelation time.
'''

import numpy as np
from collections import defaultdict

def DataT(FF):
    Obs = np.loadtxt(FF)
    NMeas = Obs.size   
    return Obs, NMeas

def DataT2(FF):
    Obs = np.loadtxt(FF)
    NMeas = Obs.size   
    Aver = np.sum(Obs) / NMeas
    Ave2 = np.sum(Obs**2) / NMeas
    Den = Ave2 - Aver**2
    return Aver, Den   


def rho_f(XX,j):
    Ob, Nm = DataT(XX)
    Np = Nm-j
    A0 = 0
    for i in range(Np):
        A0 += Ob[i]*Ob[i+j]
    Ren = A0/float(Np)
    return Ren

def AutoC(XX,CutOff=100):
    rho_t = []
    Ob, Nm = DataT(XX)
    Av, De = DataT2(XX)   
    CC = min(int(Nm), CutOff)
    tau = 1.5
    for i in range(CC):
        numerator = rho_f(XX, i) - Av**2
        if numerator > 0:
            rho_t.append( numerator / De)
            tau +=  rho_t[i]
        else:
            if abs(rho_t[i-1] - rho_t[i-2])>1e-15:
                tau += - rho_t[i-1] / (1.0 - rho_t[i-1] / rho_t[i-2])
            break
    return rho_t, tau

def mean_and_err(At,i_bins,T_list):
    '''
    Returns the mean and variance of the autocorrelation time over the i_bins simulations
    for every temperature in T_list
    '''
    At_mean = defaultdict(list)
    At_err = defaultdict(list)
    for T in T_list:
        At_mean[T] = np.mean([At[i][T] for i in i_bins],axis=0)
        At_err[T]  = np.std([At[i][T] for i in i_bins],axis=0)/np.sqrt(len(i_bins))
    return At_mean, At_err

if __name__ == "__main__":
    import numpy as np
    import random, math
    import matplotlib.pyplot as plt
    import matplotlib 
    from collections import defaultdict
    import threading
    import pickle
    
    # Computation of autocorrelation function for Metropolis algorithm
    
    N_sweeps = 20_000          # Number of steps for the measurements
    N_eq     = 10_000      # Number of equilibration steps before the measurements start 
    N_flips  = 400         # Number of steps between measurements 

    T_list = [1.,2.269,3.]          # temperatures
    L_list = [8,10,12,14,16,18,20]   # system sizes
    
    i_bins = range(10)
    
    AtM_tau_MSF = {i: {T:[0 for _ in L_list] for T in T_list} for i in i_bins}
    AtM_rho_MSF = {i: {T:[0 for _ in L_list] for T in T_list} for i in i_bins}

    cdirM = './MagnetizationSingleFlip/'
    #cdirE = "./EnergySingleFlip/"

    def compute_autoc(i,T,L):
        print(f"{i=}\t{T=} {L=}")
        L_index = L_list.index(L)
        
        # compute autocorrelation of the magnetization
        MFile = f'{cdirM}{i}-Magnetization_{L}L_{T:.3f}T_{N_sweeps}Nsw_{N_eq}Neq_{N_flips}Nfl-SingleFlip.dat'
        rho, tau = AutoC(MFile,CutOff=500)
        AtM_tau_MSF[i][T][L_index] = tau
        AtM_rho_MSF[i][T][L_index] = rho
        
        print(f"finished {i=} {T=} {L=}")
    
    print(f"Parameters:\t {N_sweeps=}\t {N_eq=}\t {N_flips=}")
    # parallel execution of the code
    
    # parallel execution of the code
    threads = []
    for T in T_list:
        for L in L_list:
            for i in i_bins:
                t = threading.Thread(target=compute_autoc,args=[i,T,L])
                t.start()
                threads.append(t)
    for thread in threads:
        thread.join()
    
    #save the results
    cdirAC = './AutoCorrelation/'

    with open(f'{cdirAC}AutoCtau-Magnetization_{N_sweeps}Nsw_{N_eq}Neq_{N_flips}Nfl-SingleFlip.pickle', 'wb') as f:
        pickle.dump(AtM_tau_MSF, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(f'{cdirAC}AutoCrho-Magnetization_{N_sweeps}Nsw_{N_eq}Neq_{N_flips}Nfl-SingleFlip.pickle', 'wb') as f:
        pickle.dump(AtM_rho_MSF, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Autocorrelation completed\nNumber of bins: {len(i_bins)}\nParameters:\t {N_sweeps=}\t {N_eq=}\t {N_flips=}")
    print(f"Temperatures: {T_list}\nSystem sizes: {L_list}")