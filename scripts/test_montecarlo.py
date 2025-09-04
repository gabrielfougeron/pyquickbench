# import os
import math
import numpy as np
import functools
import pyquickbench
import ot
import itertools

# print(ot.backend.get_available_backend_implementations())

TT = pyquickbench.TimeTrain(names_reduction='sum', include_locs=False, relative_timings = True)

# np.random.seed(seed=0)

nvec = 10
n = 4000
lenlist = [n] * nvec
# lenlist = [5, 20, 50, 100]

nmc = 10000

poc_exh = TT.tictoc(functools.partial(pyquickbench.rankstats.score_to_partial_order_count, method = "exhaustive"), name="exhaustive")
poc_mc = TT.tictoc(functools.partial(pyquickbench.rankstats.score_to_partial_order_count, method = "montecarlo", nmc = nmc), name="montecarlo")

order_th = np.array(range(nvec))

d = -0.1
k = 2

n_th_OK = 0
n_mc_OK = 0

n_repeat = 10
for i_repeat in range(n_repeat):
    
    l = [np.random.random(lenlist[ivec]) + d*ivec for ivec in range(nvec)]
    
    res_exh = poc_exh(k, l)
    A, p, q = pyquickbench.rankstats.build_sinkhorn_problem(res_exh)
    u_noreg, v_noreg = pyquickbench.cython.sinkhorn.sinkhorn_knopp(A, p, q)
    order_exh = np.argsort(v_noreg)
    
    res_mc = poc_mc(k, l)
    A, p, q = pyquickbench.rankstats.build_sinkhorn_problem(res_mc)
    u_noreg, v_noreg = pyquickbench.cython.sinkhorn.sinkhorn_knopp(A, p, q)
    order_mc = np.argsort(v_noreg)
    
    if np.array_equal(order_exh, order_th):
        n_th_OK += 1    
    if np.array_equal(order_exh, order_mc):
        n_mc_OK += 1
        
    print(order_exh)
    
print(f'{n_th_OK = } / {n_repeat}')
print(f'{n_mc_OK = } / {n_repeat}')
print()
    
    

    
    
print(TT)