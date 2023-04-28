'''
Metropolis single flip updates for the 2D Ising model. 
Save magnetizations on a file. 
'''

import numpy as np
import random
import threading
import time

# set simulation parameters

N_sweeps = 20_000          # Number of steps for the measurements
N_eq     = 10_000          # Number of equilibration steps before the measurements start 
N_flips  = 400             # Number of steps between measurements 

T_list = [1.,2.269,3.]                # temperatures
L_list = [8,10,12,14,16,18,20]        # system sizes

i_bins = range(10)

# define the main functions

def energy(system,j,L):
    N = L*L
    nbr = ((j // L) * L + (j + 1) % L, (j + L) % N, (j // L) * L + (j - 1) % L, (j - L) % N)
    """ Energy function """
    return -1. * system[j] * sum([system[k] for k in nbr])

def prepare_system(L,T):
    """ Initialize the system """
    N = L* L
    system = [random.choice([1, -1]) for k in range(N)]
    if T<1.1:
        system = [1 for k in range(N)] # cold start
    E = sum([energy(system,i,L)/2 for i in range(N)])
    return system, E

def metropolis_loop(L,T):
    PEff = 1.0-np.exp(-2. / T)
    """ Main loop doing the Metropolis algorithm """
    MM_list = []
    EM_list = []
    N = L*L
    system, E = prepare_system(L,T)
    for step in range(N_sweeps*N_flips + N_eq):
        i = np.random.randint(0,N)
        dE = -2. * energy(system,i,L)
        if dE <= 0.:
            # single spin flips
            system[i] *= -1
            E += dE
        elif np.exp(-1./T*dE) > np.random.rand():
            system[i] *= -1
            E += dE
        if step >= N_eq and step%N_flips==0:
            EM_list.append(E)
            MM_list.append(sum(system)) 
    return np.array(MM_list),np.array(EM_list)

# execute the code, extracting the magnetization list, print on file

cdirM = "./MagnetizationSingleFlip/"
cdirE = "./EnergySingleFlip/"

def execute_metropolis(i_run,L,T):                         
    print(f"{i_run=} computing... dimension of the system: {L}, temperature: {T:.3f}")
    #FileNameM = f'{cdirM}{i_run}-Magnetization_{L}L_{T:.3f}T_{N_sweeps}Nsw_{N_eq}Neq_{N_flips}Nfl-SingleFlip.dat'
    FileNameE = f'{cdirE}{i_run}-Energy_{L}L_{T:.3f}T_{N_sweeps}Nsw_{N_eq}Neq_{N_flips}Nfl-SingleFlip.dat'
    M,E = metropolis_loop(L,T)
    #np.savetxt(FileNameM, M)
    np.savetxt(FileNameE, E)
    print(f"finished: {i_run=}, dimension of the system: {L}, temperature: {T:.3f}")
    
print(f"Parameters:\t {N_sweeps=}\t {N_eq=}\t {N_flips=}")

# parallel execution of the code
threads = []
for i in i_bins:
    for L in L_list:
        for T in T_list:
            t = threading.Thread(target=execute_metropolis,args=[i,L,T])
            t.start()
            threads.append(t)
for thread in threads:
    thread.join()
    
print(f"Computation completed\nNumber of bins: {len(i_bins)}\nParameters:\t {N_sweeps=}\t {N_eq=}\t {N_flips=}")
print(f"Temperatures: {T_list}\nSystem sizes: {L_list}")
