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

nvec = 4
n = 10
lenlist = [n] * nvec
# lenlist = [5, 20, 50, 100]

# poc_opt = TT.tictoc(functools.partial(pyquickbench.rankstats.score_to_partial_order_count, opt = "opt"), name="poc_opt")
# poc_bfc = TT.tictoc(functools.partial(pyquickbench.rankstats.score_to_partial_order_count, opt = "brute_force_compiled"), name="poc_bfc")
# poc_bf = TT.tictoc(functools.partial(pyquickbench.rankstats.score_to_partial_order_count, opt = "brute_force"), name="poc_bf")

# print(lenlist)

d = 0.0
l = [np.random.random(lenlist[ivec]) + d*ivec for ivec in range(nvec)]



moy = np.array([np.mean(l[ivec]) for ivec in range(nvec)])
med = np.array([np.median(l[ivec]) for ivec in range(nvec)])

# print(moy)

# print(moy)
print(np.argsort(moy))

# print()
# print(med)
print(np.argsort(med))
# print(np.argsort(log_v))
# print()

for method in [
    'sinkhorn'  ,
    'sinkhorn_log'  ,
    'greenkhorn'    ,
    # 'sinkhorn_stabilized'   ,
    # 'sinkhorn_epsilon_scaling'   ,
]:

    print()
    print(f'{method = }')

    for k in range(2,3):
    # for k in range(2,nvec+1):
        
        order_count = pyquickbench.rankstats.score_to_partial_order_count(k, l)
        A, p, q = pyquickbench.rankstats.build_sinkhorn_problem(order_count)

        M, log = ot.bregman.sinkhorn(
            p,
            q,
            -np.log(A),
            reg = 1. ,
            method=method,
            numItermax=1000000,
            stopThr=1e-13,
            verbose=False,
            log=True,
            warn=False,
            warmstart=None,
        )

        u = log['u']
        v = log['v']
        Ah = np.einsum('i,ij,j->ij', u, A, v)
        
        print(np.linalg.norm(M-Ah))
        print(np.linalg.norm(p-np.sum(Ah,axis=1)))
        print(np.linalg.norm(q-np.sum(Ah,axis=0)))

        log_v = np.log(v)
        log_v -= np.sum(log_v) / log_v.shape[0]

        # print("sin", k, np.argsort(log_v))
        
        print(f'{log_v = }')
