# import os

import numpy as np
import pyquickbench

n = 1000

class CountCmp():

    def __init__(self):
        self.reset()
    
    def reset(self):
        self.n=0
    
    def __call__(self, a, b):
        self.n += 1
        return a < b    
    

arr = np.random.random(n)
cmp_lt = CountCmp()
