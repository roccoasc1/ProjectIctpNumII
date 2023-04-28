'''
Wolff cluster updates for the 2D Ising model. 
Save the magnetizations on a file. 
Option to start form both ordered or disordered state
'''

import numpy as np
import random, math
import threading

# Set simulation parameters

N_sweeps = 2_000        # Number of steps for the measurements
N_eq     = 1_00        # Number of equilibration steps before the measurements start 
N_flips  = 1           # Number of steps between measurements 

T_list = [1.,2.269,3.]                      # temperatures   
L_list = [8,12,16,20,24,28,32,36,40]  # system sizes

#T_list = [2.5,2.7,2.9]
#L_list = [20,40,60,80,100]

# define wolff loop
def wolff_loop(L,T):
    MW_list = []
    
    N = L * L
    # Definition of the neighbors of the site [i]. shall be changed according to BC and dimensionality

    nbr = {i : ((i // L) * L + (i + 1) % L, (i + L) % N,
                (i // L) * L + (i - 1) % L, (i - L) % N)
                                        for i in range(N)}

    p  = 1.0 - math.exp(-2.0 / T) # probability acceptance in order to satisfy detailed balance
    
    # Initial state

    #S = [random.choice([1, -1]) for k in range(N)] #hot start
    S = [1 for k in range(N)] # cold start

    def CycleWolff(Sl):
        # selects a random spin
        k = random.randint(0, N - 1)
        k_state = Sl[k]
        Sl[k] *= -1 
        Pocket = set([k])
        while Pocket != set():
            j = random.choice(list(Pocket))
            for l in nbr[j]:
                # inspects the neighbors
                if Sl[l] == k_state and random.uniform(0.0, 1.0) < p:
                    # spin accepted
                    Sl[l] *= -1
                    Pocket.add(l)
            Pocket.remove(j)
        return Sl
    
    # Equilibration
    for step in range(N_eq):
        S = CycleWolff(S)

    # Sampling
    for step in range(N_sweeps):
        # cycle steps to get uncorrelated data
        for _ in range(N_flips):
            S = CycleWolff(S)
        MW_list.append(sum(S))
        
    return np.array(MW_list)

# execute the code, extracting the magnetization list, print on file

cdir="./MagnetizationWolff/"

def execute_wolff(i_run,L,T):
    print(f"{i_run=} computing... dimension of the system: {L}, temperature: {T:.3f}")
    FileNameM = f'{cdir}{i_run}-Magnetization_{L}L_{T:.3f}T_{N_sweeps}Nsw_{N_eq}Neq_{N_flips}Nfl-Wolff.dat'
    M = wolff_loop(L,T)
    np.savetxt(FileNameM, M)
    print(f"finished: {i_run=}, dimension of the system: {L}, temperature: {T:.3f}")

nbins = 10
i_bins = range(nbins)

print(f"Parameters:\t {N_sweeps=}\t {N_eq=}\t {N_flips=}")

# parallel execution of the code
threads = []
for i in i_bins:
    for L in L_list:
        for T in T_list:
            t = threading.Thread(target=execute_wolff,args=[i,L,T])
            t.start()
            threads.append(t)
for thread in threads:
    thread.join()
    
print(f"Computation completed\nNumber of bins: {len(i_bins)}\nParameters:\t {N_sweeps=}\t {N_eq=}\t {N_flips=}")
print(f"Temperatures: {T_list}\nSystem sizes: {L_list}")