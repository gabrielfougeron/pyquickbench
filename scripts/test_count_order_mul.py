# import os
import math
import numpy as np
import functools
import pyquickbench
import ot
import itertools
import scipy

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

        method = 'sinkhorn'

        M, log = ot.bregman.sinkhorn(
            p                   ,
            q                   ,
            -np.log(A)          ,
            reg = 1.            ,
            method=method       ,
            numItermax=1000000  ,
            stopThr=1e-13       ,
            verbose=False       ,
            log=True            ,
            warn=False          ,
            warmstart=None      ,
        )

        u = log['u']
        v = log['v']

        # print(np.linalg.norm(p-np.sum(M,axis=1)))
        # print(np.linalg.norm(q-np.sum(M,axis=0)))
                
        Ah = np.einsum('i,ij,j->ij', u, A, v)
        
        # print(np.linalg.norm(np.matmul(A  , v) - p/u))
        # print(np.linalg.norm(np.matmul(A.T, u) - q/v))

        log_v = np.log(log['v'])
        log_v -= np.sum(log_v) / log_v.shape[0]
        
        # print(log_v)

        # print(np.argsort(log_v))
        print("sin", k, np.argsort(log_v))

        # print(np.sort(log_v))
        
        J = pyquickbench.rankstats.build_tangent_sinkhorn_problem(A, u, v)
        
        # print(J)
        

# 
#         print(np.linalg.norm(M-Ah))
#         print(np.linalg.norm(p-np.sum(Ah,axis=1)))
#         print(np.linalg.norm(q-np.sum(Ah,axis=0)))
#         print(np.linalg.norm(J))
#         
        # U, s, Vh = scipy.linalg.svd(J)
        
        # for i in range(s.shape[0]):
        #     if abs(s[i]) < 1e-10:
        #         s[i] = 0
        
        # print(s)
        
        
        d_order_count = np.zeros_like(order_count)
        d_order_count[0,0] = 1
        d_order_count[0,1] = -1

        A, pp, qp = pyquickbench.rankstats.build_sinkhorn_problem(order_count+d_order_count)
        
        dp = pp - p
        dq = qp - q
        
        dpq = np.concatenate((dp,dq))
        
        # print(f'{dpq = }')
        
        eps = 1
        
        Mp, logp = ot.bregman.sinkhorn(
            p+eps*dp            ,
            q+eps*dq            ,
            -np.log(A)          ,
            reg = 1.            ,
            method=method       ,
            numItermax=1000000  ,
            stopThr=1e-13       ,
            verbose=False       ,
            log=True            ,
            warn=False          ,
            warmstart=None      ,
        )   
          
        Mm, logm = ot.bregman.sinkhorn(
            p-eps*dp            ,
            q-eps*dq            ,
            -np.log(A)          ,
            reg = 1.            ,
            method=method       ,
            numItermax=1000000  ,
            stopThr=1e-13       ,
            verbose=False       ,
            log=True            ,
            warn=False          ,
            warmstart=None      ,
        )
        
        
        du = (logp['u'] - logm['u']) / (2*eps)
        dv = (logp['v'] - logm['v']) / (2*eps)
        
        duv_diff_fin = np.concatenate((du,dv))
        
        # duv , _ , _ , _= np.linalg.lstsq(J,dpq)
        duv = np.linalg.solve(J,dpq)
        
        print(duv)
        print(duv_diff_fin)
        # print(duv_diff_fin - duv)
        # 
        # print(np.linalg.norm(duv_diff_fin - duv) / np.linalg.norm(duv))
        
        
        
        
        