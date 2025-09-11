# import os
import math
import numpy as np
import functools
import pyquickbench
import scipy

# print(ot.backend.get_available_backend_implementations())

# TT = pyquickbench.TimeTrain(names_reduction='sum', include_locs=False, relative_timings = True)

# np.random.seed(seed=0)

eps = 1e-14

nvec = 4
n = 10000
lenlist = [1000, 1000, 1000, 1000]

dlist = [0.0, 0.0, 0.333333333333333, 0.66666666666666]
# dlist = [0.0, 0.666666666666666666666666666666]
k = 2

l = [np.random.random(lenlist[ivec]) + dlist[ivec] for ivec in range(nvec)]
nmc_all = np.array([10, 1000, 1000, 1000, 1000, 1000])
# nmc_all = np.ones(6, dtype=np.intp) * 100000

# order_count = pyquickbench.rankstats.score_to_partial_order_count(k, l)
order_count = pyquickbench.rankstats.score_to_partial_order_count(k, l, method="montecarlo", nmc_all = nmc_all)
assert np.array_equal(nmc_all, order_count.sum(axis=1))


# A, p, q = pyquickbench.rankstats.build_sinkhorn_problem(order_count)
A, p, q, dq = pyquickbench.rankstats.build_sinkhorn_problem_2(order_count, reg_eps = 0.00000)

order_count_best = pyquickbench.rankstats.project_order_count_best(order_count)
pp, qq, dqq = pyquickbench.rankstats.build_sinkhorn_rhs(order_count_best, reg_eps = 0.00000)

assert np.linalg.norm(pp-p) < eps
assert np.linalg.norm(qq-q) < eps
assert np.linalg.norm(dqq-dq) < eps






reg_beta = 0.
reg_alpham1 = 0.
# reg_beta = 0.1
# reg_alpham1 = 0.1

u, v = pyquickbench.cython.sinkhorn.sinkhorn_knopp(A, p, q, reg_alpham1 = reg_alpham1, reg_beta = reg_beta)
M = np.einsum('i,ij,j->ij',u,A,v)
J = pyquickbench.rankstats.build_log_tangent_sinkhorn_problem(M)

eigvals, eigvects = scipy.linalg.eigh(J)

err = np.linalg.norm( J - np.matmul(eigvects, np.matmul(np.diag(eigvals), eigvects.T)))
print(err)
assert err < eps


nsets = order_count.shape[0]

# uprod = np.prod(u)
uprod = 1.

uscale = u
# uscale = u /(order_count.sum(axis=1)/order_count.sum())
# uscale = u /(order_count.sum(axis=1)/order_count.sum()) / (uprod**(1/nsets))

print(p)
print(q)
print()
logu, logv = pyquickbench.cython.sinkhorn.uv_to_loguv(u, v)

# lam = (logv.sum() - logu.sum()) / (nsets+nvec)
# logu += lam
# logv -= lam

print(logu)
print(logv)
print(logv.sum() - logu.sum())
print()
print(M)
# 
# print()
# print(v[0] / (v[0] + v[1]))
# print(1/2)
# 
# print()
# print(v[0] / (v[0] + v[2]))
# print(2/9)
# 
# print()
# print(v[0] / (v[0] + v[3]))
# print(1/18)

# print()
# print(v[0] / (v[0] + v[1]))
# print(1/18)


for iset in range(nsets):
    print(iset, pyquickbench.rankstats.unrank_combination(iset,nvec,k), uscale[iset], p[iset])


print(v)
print()

# print(np.matmul(J, eigvects) - np.matmul(eigvects, np.diag(eigvals)))


print(eigvals)
# print(eigvects[:,0])
# print(eigvects[:,1])
# print(eigvects[:,2])


n_tests_done = order_count.sum(axis=1) + 1


for iset in range(nsets):
    
    dpq = np.zeros((nsets+nvec), dtype=np.float64)
    
    dpq[iset] = 1. / n_tests_done[iset]
    dpq[nsets:] = dq[iset,:] / n_tests_done[iset]
    
    dloguv , _ , _ , _= np.linalg.lstsq(J,dpq)
    
    
    
    
    du = u*dloguv[:nsets]
    dv = v*dloguv[nsets:]
    
    print()
    print(iset, pyquickbench.rankstats.unrank_combination(iset,nvec,k))
    # print(dloguv[:nsets])
    # print(dloguv[nsets:])
    # print(dv)
    print(np.linalg.norm(dv))