'''
Wolff cluster updates for the 2D Ising model. 
Save the spin configuration on a file. 
'''

import numpy as np
import random, math
import threading

L_list = [20,40,60,80,100]
T_list = np.arange(1.6,3.,0.1)  # temperatures


nwup = 2_000      # Number of equilibration steps before the measurements start
m_measure = 100   # Number of steps for the measurements

L_dict_nc = {20:20, 40:60, 60:100, 80:150, 100:200} # Number of steps between measurements as function of system size {L:nc,..}

cdir = './SpinConfiguration/'

def get_spin_config(L,nc,T):
    print(f"computing... dimension of the system: {L}, temperature: {T:.1f}")

    N = L * L

    # definition of the neighbors of the site [i]. shall be changed according to BC and dimensionality

    nbr = {i : ((i // L) * L + (i + 1) % L, (i + L) % N,
                (i // L) * L + (i - 1) % L, (i - L) % N) for i in range(N)}

    p  = 1.0 - math.exp(-2.0 / T)

    # create file to append spin configurations
    FileNameSC = f'{cdir}spin_config_{L}L_{T:.1f}T_{nc}flips.dat'
    with open(FileNameSC,'w') as f:
        f.write('') # reset file

    # Initial state
    S = [1 for k in range(N)]

    def Cycle(Sl):
            # selects a random spin
            k = random.randint(0, N - 1)
            Pocket, Cluster = set([k]), set([k])
            while Pocket != set():
                j = random.choice(list(Pocket))
                for l in nbr[j]:
                    # inspects the neighbors
                    if Sl[l] == Sl[j] and l not in Cluster and random.uniform(0.0, 1.0) < p:
                        Pocket.add(l)
                        Cluster.add(l)
                Pocket.remove(j)
            for j in Cluster:
                Sl[j] *= -1
            return Sl

    # Warm up
    for step in range(nwup):
        S = Cycle(S)

    # Sampling
    for step in range(m_measure):
        for j in range(nc):
            S = Cycle(S)
        # save the spin configuration
        with open(FileNameSC,'a+') as f:
            for s in S:
                f.write(f"{s} ")
            f.write("\n")
    print(f"finished: dimension of the system: {L}, temperature: {T:.1f}")
        
# parallel execution of the code
threads = []
for L, nc in L_dict_nc.items():
    for T in T_list:
        t = threading.Thread(target=get_spin_config,args=[L,nc,T])
        t.start()
        threads.append(t)
for thread in threads:
    thread.join()

print(f"Computation completed\n")
print(f"Temperatures {T_list}\nSystem size: N_filps {L_dict_nc}")