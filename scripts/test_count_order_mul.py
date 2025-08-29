# import os
import math
import numpy as np
import functools
import pyquickbench
import ot
import itertools

# print(ot.backend.get_available_backend_implementations())

TT = pyquickbench.TimeTrain(names_reduction='avg', include_locs=False)

# np.random.seed(seed=0)

nvec = 6
n = 11
lenlist = [n] * nvec
# lenlist = [3,4,5,6,7,8]
n_repeat = 10

for i_repeat in range(n_repeat):
    
    print()
    print(f'{i_repeat = }')

    d = 0.
    l = [np.random.random(lenlist[ivec]) + d*ivec for ivec in range(nvec)]

    
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
        
        # print(q)
        
        # print("qin", k,np.argsort(q))

        # print(order_count.sum())

        method = 'sinkhorn'

        M, log = ot.bregman.sinkhorn(
            p,
            q,
            A,
            reg = 1. ,
            method=method,
            numItermax=1000000,
            stopThr=1e-13,
            verbose=False,
            log=True,
            warn=False,
            warmstart=None,
        )
        
        # print(log['v'].sum())

        log_v = np.log(log['v'])
        log_v -= np.sum(log_v) / log_v.shape[0]
        
        # print(log_v)

        # print(np.argsort(log_v))
        print("sin", k, np.argsort(log_v))
        
        # assert np.array_equal(np.argsort(q),np.argsort(log_v))
        
        # print()
        
        # print(np.sort(log_v))
