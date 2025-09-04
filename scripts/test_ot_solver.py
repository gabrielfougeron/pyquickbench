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

TT = pyquickbench.TimeTrain(names_reduction='sum', include_locs=False)

# np.random.seed(seed=0)

nvec = 4
# n = 1000
# lenlist = [n] * nvec
# lenlist = [10, 10, 10]
lenlist = [2,3,5,7]
n_repeat = 1000


# method = 'sinkhorn'
# method = 'sinkhorn_log'
# method = 'greenkhorn'
# method = 'sinkhorn_stabilized'
# method = 'sinkhorn_epsilon_scaling

methods = [
    'sinkhorn',
    # 'sinkhorn_log',
    # 'greenkhorn',
    # 'sinkhorn_stabilized',
    # 'sinkhorn_epsilon_scaling',
]

stopThr = 1e-6
neq = 0
for i_repeat in range(n_repeat):
# for i_repeat in [8]:
    
    print()
    print(f'{i_repeat = }')
    
    np.random.seed(seed=i_repeat)

    d = 0.
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

        print()
        print(f'{k = }')    
        
        order_count = pyquickbench.rankstats.score_to_partial_order_count(k, l)
        A, p, q = pyquickbench.rankstats.build_sinkhorn_problem(order_count)
        
        nsets = p.shape[0]
        nopts = q.shape[0]
        
        n = nsets+nopts

        TT.toc("start")
        u_noreg, v_noreg = pyquickbench.cython.sinkhorn.sinkhorn_knopp(
            p                   ,
            q                   ,
            A                   ,
            # reg = 1.            ,
            numItermax=10000000  ,
            stopThr=stopThr       ,
            warmstart=None      ,
        )
        TT.toc("Cython noreg")
        
        # print("u_noreg err ", np.linalg.norm(p-u_noreg*np.dot(A,v_noreg)))
        # print("v_noreg err ", np.linalg.norm(q-np.dot(u_noreg,A)*v_noreg))  
        
        reg_alpham1 = 1e-4
        reg_beta = 1.e-4
        
        TT.toc("start")
        u_reg, v_reg = pyquickbench.cython.sinkhorn.sinkhorn_knopp(
            p                           ,
            q                           ,
            A                           ,
            reg_alpham1 = reg_alpham1   ,
            reg_beta = reg_beta         ,
            numItermax=10000000           ,
            stopThr=stopThr               ,
            warmstart=None              ,
        )
        TT.toc("Cython reg")
        
        # print("u_reg err ", np.linalg.norm(p-u_reg*np.dot(A,v_reg)))
        # print("v_reg err ", np.linalg.norm(q-np.dot(u_reg,A)*v_reg))        
        # print("v_reg err ", np.linalg.norm((q+reg_alpham1)-(np.dot(u_reg,A)+reg_beta)*v_reg))        
        
        # assert np.linalg.norm(p-ucy*np.dot(A,vcy)) < 1e-13
        # assert np.linalg.norm(q-np.dot(ucy,A)*vcy) < 1e-13
        
        # print(np.argsort(v_noreg))
        # print(np.argsort(v_reg))
        
        if np.array_equal(np.argsort(v_noreg), np.argsort(v_reg)):
            neq += 1
        
print()
print()
print()
print(f'{neq} / {(nvec-1)*n_repeat}')
print()

print(TT)