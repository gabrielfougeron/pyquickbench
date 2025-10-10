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

eps = 1e-14

nvec = 4
n = 10000
lenlist = [1000, 1000, 1000, 1000]

dlist = [0.0, 0.0, 0.333333333333333, 0.66666666666666]
k = 2

l = [np.random.random(lenlist[ivec]) + dlist[ivec] for ivec in range(nvec)]
nmc_all = np.array([10, 1000, 1000, 1000, 1000, 1000])

# order_count = pyquickbench.rankstats.score_to_partial_order_count(k, l)
order_count = pyquickbench.rankstats.score_to_partial_order_count(k, l, method="montecarlo", nmc_all = nmc_all)

project_order_count_best = TT.tictoc(pyquickbench.rankstats.project_order_count_best, name="project_order_count_best")
build_sinkhorn_problem_order = TT.tictoc(pyquickbench.rankstats.build_sinkhorn_problem_order, name="build_sinkhorn_problem_order")


n_repeat = 100000
for i_repeat in range(n_repeat):

    order_count_best = project_order_count_best(order_count)
    A, p, q = build_sinkhorn_problem_order(order_count, reg_eps = 0., minimize=False)

sum_tot = order_count_best.sum()
pp = order_count_best.sum(axis=1) / sum_tot
qq = order_count_best.sum(axis=0) / sum_tot

assert (np.linalg.norm(p-pp)) < eps
assert (np.linalg.norm(q-qq)) < eps


print(TT)