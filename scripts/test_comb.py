# import os
import math
import numpy as np
import functools
import pyquickbench
import ot
import itertools

i = 250
n = 8


# print(math.comb(i,n))
# print(pyquickbench.cython.rankstats.binomial(i,n))
# 
# 
# 
# fac = pyquickbench.cython.rankstats.to_factorial_base(i,n)
# print(fac)
# 
# perm = pyquickbench.cython.rankstats.from_left_lehmer(i,n)
# print(perm)
# 
# 
# for i, perm in enumerate(itertools.permutations(range(8))):
#     
#     print(i,perm)
#     
#     if i > 2026:
#         break
#     

n = 8

for iset, comb in enumerate(itertools.combinations(range(n), 2)):
    
    jset = comb[1]-1-(comb[0]+3-2*n)*comb[0]//2
    print(iset, jset)
    
    assert iset == jset
    