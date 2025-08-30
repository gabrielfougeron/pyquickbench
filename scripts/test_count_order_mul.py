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
lenlist = [10, 100, 1000]
n_repeat = 1


# method = 'sinkhorn'
method = 'sinkhorn_log'
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

        log_v = np.log(log['v'])
        log_u = np.log(log['u'])
        dd = (np.sum(log_v)-np.sum(log_u)) / n
        log_v -= dd
        log_u += dd

        print("sin", k, np.argsort(log_v))
        
        # print(f'{log_v = }')

        # print(np.argsort(log_v))


        # print(np.sort(log_v))
        
        J = pyquickbench.rankstats.build_log_tangent_sinkhorn_problem(M)
        
        
        # print(np.linalg.norm(p-u*np.dot(A,v)))
        # print(np.linalg.norm(q-np.dot(u,A)*v))

        
        d_order_count = pyquickbench.rankstats.score_to_partial_order_count(k, dl)
        A, pp, qp = pyquickbench.rankstats.build_sinkhorn_problem(d_order_count)
        
        dp = p - pp
        dq = q - qp
        
        # dp = np.random.random(nsets)
        # dq = np.random.random(nopts)
        # dp -= dp.sum()/nsets
        # dq -= dq.sum()/nopts
        
        dpq = np.concatenate((dp,dq))
        
        dloguv , _ , _ , _= np.linalg.lstsq(J,dpq)
        
        # print(f'{dpq = }')
        
        neps = 10
        errs = np.empty((neps), dtype=np.float64)
        
        for ieps in range(neps):
            
            # print(ieps)
            
            eps = 10**(-ieps)
            
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
            
            log_vp = np.log(logp['v'])
            log_up = np.log(logp['u'])
                    
            dd = (np.sum(log_vp) - np.sum(log_up) ) / n
            log_vp -= dd
            log_up += dd

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
            
            log_vm = np.log(logm['v'])
            log_um = np.log(logm['u'])
            
            dd = (np.sum(log_vm) - np.sum(log_um) ) / n
            log_vm -= dd
            log_um += dd
            
            
            dlogu = (log_up - log_um) / (2*eps)
            dlogv = (log_vp - log_vm) / (2*eps)
            
            duv_diff_fin = np.concatenate((dlogu,dlogv))
            
            relerr = np.linalg.norm(dloguv - duv_diff_fin) / np.linalg.norm(dloguv)
            
            errs[ieps] = relerr
            
        # print(errs.min())
            
            
        
        
        