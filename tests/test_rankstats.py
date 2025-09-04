import os
import sys

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import pytest
import warnings
from test_config import *

import numpy as np
import pyquickbench

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
    [100]           ,
    [10]*3          ,
    [10,20,30,35]   ,
]

lenlist_illcond_list = [
    [2,3,4,5]   ,
]

@pytest.mark.parametrize("lenlist", lenlist_list)
def test_exhaustive_score_to_partial_order_count(lenlist):
    
    nvec = len(lenlist)
    l = [np.random.random(lenlist[ivec]) for ivec in range(nvec)]

    for k in range(1,nvec+1):

        poc_opt = pyquickbench.rankstats.exhaustive_score_to_partial_order_count(k, l, opt="opt")
        poc_bf =  pyquickbench.rankstats.exhaustive_score_to_partial_order_count(k, l, opt="brute_force")
        
        assert np.array_equal(poc_opt, poc_bf)

@pytest.mark.parametrize("lenlist", lenlist_list)
def test_sinkhorn_solver(lenlist, reltol=1e-8):

    nvec = len(lenlist)
    l = [np.random.random(lenlist[ivec]) for ivec in range(nvec)]

    for k in range(2,nvec+1):
        
        order_count = pyquickbench.rankstats.score_to_partial_order_count(k, l)
        A, p, q = pyquickbench.rankstats.build_sinkhorn_problem(order_count)
        
        u, v = pyquickbench.cython.sinkhorn.sinkhorn_knopp(
            A, p, q,
            stopThr = 1e-13 ,
        )
        
        assert np.linalg.norm(p-u*np.dot(A,v)) < reltol
        assert np.linalg.norm(q-np.dot(u,A)*v) < reltol
        
        assert np.all(p >= 0)
        assert np.all(q >= 0)

@pytest.mark.parametrize("lenlist", lenlist_illcond_list)
def test_sinkhorn_solver_illcond(lenlist, reltol=1e-8):

    reg_alpham1 = 0.00001
    reg_beta = 0.00001

    nvec = len(lenlist)
    l = [np.random.random(lenlist[ivec]) for ivec in range(nvec)]

    for k in range(2,nvec+1):
        
        order_count = pyquickbench.rankstats.score_to_partial_order_count(k, l)
        A, p, q = pyquickbench.rankstats.build_sinkhorn_problem(order_count)
        
        u, v = pyquickbench.cython.sinkhorn.sinkhorn_knopp(
            A, p, q,
            reg_alpham1 = reg_alpham1   ,
            numItermax = 1000000        ,
            reg_beta = reg_beta         ,
            stopThr = 1e-13             ,
        )
        
        assert np.linalg.norm(p-u*np.dot(A,v)) < reltol
        assert np.linalg.norm((q+reg_alpham1)-(np.dot(u,A)+reg_beta)*v) < reltol
        
        assert np.all(p >= 0)
        assert np.all(q >= 0)
        
@pytest.mark.parametrize("lenlist", lenlist_list)
def test_luce_gradient(lenlist, reltol=1e-8):

    nvec = len(lenlist)
    l = [np.random.random(lenlist[ivec]) for ivec in range(nvec)]
    lp = [np.random.random(lenlist[ivec]) for ivec in range(nvec)]

    for k in range(2,nvec+1):
        
        order_count = pyquickbench.rankstats.score_to_partial_order_count(k, l)
        A, p, q = pyquickbench.rankstats.build_sinkhorn_problem(order_count)
        
        nsets = p.shape[0]
        nopts = q.shape[0]
        
        n = nsets+nopts

        u, v = pyquickbench.cython.sinkhorn.sinkhorn_knopp(
            A, p, q,
            stopThr = 1e-13 ,
        )

        log_v = np.log(u)
        log_u = np.log(v)
        dd = (np.sum(log_v)-np.sum(log_u)) / n
        log_v -= dd
        log_u += dd

        M = np.einsum('i,ij,j->ij',u,A,v)
        J = pyquickbench.rankstats.build_log_tangent_sinkhorn_problem(M)
        
        order_countp = pyquickbench.rankstats.score_to_partial_order_count(k, lp)
        A, pp, qp = pyquickbench.rankstats.build_sinkhorn_problem(order_countp)
        
        dp = p
        dq = q
        
        dpq = np.concatenate((dp,dq))
        
        dloguv , _ , _ , _= np.linalg.lstsq(J,dpq)

        neps = 5
        start_exp = -3
        errs = np.empty((neps), dtype=np.float64)
        
        for ieps in range(neps):
            
            eps = 10**(start_exp-ieps)
            
            up, vp = pyquickbench.cython.sinkhorn.sinkhorn_knopp(
                A                       ,
                p+eps*dp                ,
                q+eps*dq                ,
                numItermax = 1000000    ,
                stopThr = 1e-13         ,
            )
            
            assert np.linalg.norm(p+eps*dp-up*np.dot(A,vp)) < reltol
            assert np.linalg.norm(q+eps*dq-np.dot(up,A)*vp) < reltol
            
            log_vp = np.log(vp)
            log_up = np.log(up)
                    
            dd = (np.sum(log_vp) - np.sum(log_up) ) / n
            log_vp -= dd
            log_up += dd

            um, vm = pyquickbench.cython.sinkhorn.sinkhorn_knopp(
                A                       ,
                p-eps*dp                ,
                q-eps*dq                ,
                numItermax = 1000000    ,
                stopThr = 1e-13         ,
            )
                    
            assert np.linalg.norm(p-eps*dp-um*np.dot(A,vm)) < reltol
            assert np.linalg.norm(q-eps*dq-np.dot(um,A)*vm) < reltol
            
            log_vm = np.log(vm)
            log_um = np.log(um)

            dd = (np.sum(log_vm) - np.sum(log_um) ) / n
            log_vm -= dd
            log_um += dd
            
            dlogu = (log_up - log_um) / (2*eps)
            dlogv = (log_vp - log_vm) / (2*eps)
            
            duv_diff_fin = np.concatenate((dlogu,dlogv))
            
            relerr = np.linalg.norm(dloguv - duv_diff_fin) / np.linalg.norm(dloguv)
            
            errs[ieps] = relerr
            
        assert np.nanmin(errs) < reltol
                