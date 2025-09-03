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

nvec = 2
n = 400
lenlist = [n] * nvec
# lenlist = [5, 20, 50, 100]

poc_opt = TT.tictoc(functools.partial(pyquickbench.rankstats.exhaustive_score_to_partial_order_count, opt="opt"), name="poc_opt")
poc_bf_np = TT.tictoc(functools.partial(pyquickbench.rankstats.exhaustive_score_to_partial_order_count, argsort=np.argsort, opt="brute_force"), name="poc_bf_np")
poc_bf_pqb = TT.tictoc(functools.partial(pyquickbench.rankstats.exhaustive_score_to_partial_order_count, argsort=pyquickbench.cython.rankstats.insertion_argsort, opt="brute_force"), name="poc_bf_pqb")

d = 0.0
l = [np.random.random(lenlist[ivec]) + d*ivec for ivec in range(nvec)]

k = nvec

n_repeat = 1
for i_repeat in range(n_repeat):
    res_opt = poc_opt(k, l)
    res_bf_np = poc_bf_np(k, l)
    res_bf_pqb = poc_bf_pqb(k, l)

# print(res_opt)
# print(res_binary)

assert np.array_equal(res_opt, res_bf_np)
assert np.array_equal(res_opt, res_bf_pqb)

print(TT)