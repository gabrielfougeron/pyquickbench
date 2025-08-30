import os
import sys

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import pytest
import warnings
from test_config import *

import numpy as np
import pyquickbench
import ot

def test_factorial_base():

    n = 10
    
    for i in range(1000):
        
        digits = pyquickbench.cython.rankstats.to_factorial_base(i, n)
        ii = pyquickbench.cython.rankstats.from_factorial_base(digits)
        
        assert i == ii
        
        perm = pyquickbench.cython.rankstats.from_left_lehmer(i,n)
        ii == pyquickbench.cython.rankstats.left_lehmer(perm)
        
        assert i == ii

lenlist_list = [
    [100]       ,
    [10]*3      ,
    [2,3,5,7]   ,
]

@pytest.mark.parametrize("lenlist", lenlist_list)
def test_score_to_partial_order_count(lenlist):
    
    nvec = len(lenlist)
    l = [np.random.random(lenlist[ivec]) for ivec in range(nvec)]

    for k in range(1,nvec+1):

        poc_opt = pyquickbench.rankstats.score_to_partial_order_count(k, l, opt="opt")
        # poc_bfc =  pyquickbench.rankstats.score_to_partial_order_count(k, l, opt="brute_force_compiled")
        poc_bf =  pyquickbench.rankstats.score_to_partial_order_count(k, l, opt="brute_force")
        
        # assert np.array_equal(poc_opt, poc_bfc)
        assert np.array_equal(poc_opt, poc_bf )

@pytest.mark.parametrize("lenlist", lenlist_list)
def test_luce_gradient(lenlist, reltol=1e-8):
    
    # method = 'sinkhorn'
    method = 'sinkhorn_log'
    # method = 'greenkhorn'
    # method = 'sinkhorn_stabilized'
    # method = 'sinkhorn_epsilon_scaling
    
    nvec = len(lenlist)
    l = [np.random.random(lenlist[ivec]) for ivec in range(nvec)]
    lp = [np.random.random(lenlist[ivec]) for ivec in range(nvec)]

    for k in range(2,nvec+1):
        
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

        log_v = np.log(u)
        log_u = np.log(v)
        dd = (np.sum(log_v)-np.sum(log_u)) / n
        log_v -= dd
        log_u += dd

        J = pyquickbench.rankstats.build_log_tangent_sinkhorn_problem(M)
        
        order_countp = pyquickbench.rankstats.score_to_partial_order_count(k, lp)
        A, pp, qp = pyquickbench.rankstats.build_sinkhorn_problem(order_countp)
        
        dp = p - pp
        dq = q - qp
        
        dpq = np.concatenate((dp,dq))
        
        dloguv , _ , _ , _= np.linalg.lstsq(J,dpq)

        neps = 5
        start_exp = -3
        errs = np.empty((neps), dtype=np.float64)
        
        for ieps in range(neps):
            
            eps = 10**(start_exp-ieps)
            
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
            
        assert np.nanmin(errs) < reltol
            