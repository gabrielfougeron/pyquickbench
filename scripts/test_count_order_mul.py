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

nvec = 3
n = 10
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
        
        nsets = p.shape[0]
        nopts = q.shape[0]
        
        n = nsets+nopts

        # method = 'sinkhorn'
        method = 'sinkhorn_log'
        # method = 'greenkhorn'
        # method = 'sinkhorn_stabilized'
        # method = 'sinkhorn_epsilon_scaling

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
                
        # Ah = np.einsum('i,ij,j->ij', u, A, v)
        
        # print(np.linalg.norm(np.matmul(A  , v) - p/u))
        # print(np.linalg.norm(np.matmul(A.T, u) - q/v))

        log_v = np.log(log['v'])
        log_u = np.log(log['u'])
        dd = (np.sum(log_v)-np.sum(log_u)) / n
        log_v -= dd
        log_u += dd

        print("sin", k, np.argsort(log_v))
        
        print(f'{log_v = }')

        # print(np.argsort(log_v))


        # print(np.sort(log_v))
        
        J = pyquickbench.rankstats.build_log_tangent_sinkhorn_problem(M)
        
        # print(J)
        

# # 
#         print(np.linalg.norm(M-Ah))
#         print(np.linalg.norm(p-np.sum(Ah,axis=1)))
#         print(np.linalg.norm(q-np.sum(Ah,axis=0)))
        
        print(np.linalg.norm(p-u*np.dot(A,v)))
        print(np.linalg.norm(q-np.dot(u,A)*v))
        
        
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
        # 
        # print(np.sum(pp))
        # print(np.sum(qp))
        # 
        # dp = p - pp
        # dq = q - qp
        
        dp = np.random.random(nsets)
        dq = np.random.random(nopts)
        
        dp -= dp.sum()/nsets
        dq -= dq.sum()/nopts
        
        
        dpq = np.concatenate((dp,dq))
        
        # print(f'{dpq = }')
        
        eps = 1e-2
        
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
        
        # print(log_up)
        # print(log_um)
        # print(log_u)
        
        dlogu = (log_up - log_um) / (2*eps)
        dlogv = (log_vp - log_vm) / (2*eps)
        
        duv_diff_fin = np.concatenate((dlogu,dlogv))
        
        dloguv , _ , _ , _= np.linalg.lstsq(J,dpq)
        # dd = (np.sum(dloguv[0:nsets]) - np.sum(dloguv[nsets:n]) ) / n
        # dloguv -= dd
        # dloguv += dd
        
        # dd = (np.sum(dloguv[0:nsets]) - np.sum(dloguv[nsets:n]) ) / n
        # print(f'{dd = }')
        
        # dloguv = np.linalg.solve(J,dpq)
        
        # Jinv = np.linalg.inv(J)
        # print(Jinv)
        
        # dloguv = duv / np.concatenate((u,v))
        
        
        
        # print(duv[nsets:])
        # print(dlogv)
        # print(duv_diff_fin - duv)
        
        # print(dloguv)
        # print(duv_diff_fin)
        # print(dloguv - duv_diff_fin)
        
        
        # print(dlogv)
        
        print('err', np.linalg.norm(dloguv - duv_diff_fin) / np.linalg.norm(dloguv))
        
        
        
        
        