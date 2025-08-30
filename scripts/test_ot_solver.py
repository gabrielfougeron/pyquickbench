# import os
import math
import numpy as np
import functools
import pyquickbench
import ot
import itertools
import scipy

import warnings
warnings.filterwarnings("ignore")

# print(ot.backend.get_available_backend_implementations())

TT = pyquickbench.TimeTrain(names_reduction='avg', include_locs=False)

# np.random.seed(seed=0)

nvec = 3
# n = 1000
# lenlist = [n] * nvec
lenlist = [10, 10, 10]
n_repeat = 1000


method = 'sinkhorn'
# method = 'sinkhorn_log'
# method = 'greenkhorn'
# method = 'sinkhorn_stabilized'
# method = 'sinkhorn_epsilon_scaling


for i_repeat in range(n_repeat):
    
    print()
    print(f'{i_repeat = }')

    d = 0.01
    l = [np.random.random(lenlist[ivec]) + d*ivec for ivec in range(nvec)]
    
    dl = [np.random.random(lenlist[ivec]) + d*ivec for ivec in range(nvec)]
    
    order_count = pyquickbench.rankstats.score_to_partial_order_count(2, l)
    condorcet_order = pyquickbench.rankstats.condorcet_top_order(order_count)

    print("con", 0,condorcet_order)

    moy = np.array([np.mean(l[ivec]) for ivec in range(nvec)])
    med = np.array([np.median(l[ivec]) for ivec in range(nvec)])
    
    std = np.array([np.std(l[ivec]) for ivec in range(nvec)])

    # print("std", std)
    # print("std", np.argsort(std))

    print("avg", 0,  np.argsort(moy))
    print("med", 0, np.argsort(med))


    for k in range(2,nvec+1):
    # for k in range(2,3):
        
        order_count = pyquickbench.rankstats.score_to_partial_order_count(k, l)
        A, p, q = pyquickbench.rankstats.build_sinkhorn_problem(order_count)
        
        nsets = p.shape[0]
        nopts = q.shape[0]
        
        n = nsets+nopts

        
        TT.toc("start")
        M, log = ot.bregman.sinkhorn(
            p                   ,
            q                   ,
            -np.log(A)          ,
            reg = 1.            ,
            method=method       ,
            numItermax=1000  ,
            stopThr=1e-13       ,
            verbose=False       ,
            log=True            ,
            warn=False          ,
            warmstart=None      ,
        )
        TT.toc("POT")
        

        uot = log['u']
        vot = log['v']

        print("vot err ", np.linalg.norm(p-uot*np.dot(A,vot)))
        print("uot err ", np.linalg.norm(q-np.dot(uot,A)*vot))

        TT.toc("start")
        ucy, vcy = pyquickbench.cython.sinkhorn.sinkhorn_knopp(
            p                   ,
            q                   ,
            A                   ,
            # reg = 1.            ,
            numItermax=1000  ,
            stopThr=1e-13       ,
            warmstart=None      ,
        )
        TT.toc("Cython")
        
        print("vcy err ", np.linalg.norm(p-ucy*np.dot(A,vcy)))
        print("ucy err ", np.linalg.norm(q-np.dot(ucy,A)*vcy))        
        
    
print(TT)