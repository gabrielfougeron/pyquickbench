import collections
import itertools
import math
import numpy as np
import scipy

from . import rankstats

from pyquickbench._benchmark import run_benchmark
from pyquickbench._defaults import *
from pyquickbench._utils import *

class ManualRankAssign():

    def __init__(
        self                    ,
        bench_root              ,
        compare_intent  = {}    ,
        k = 2                   ,
    ):
        
        self.bench_root = bench_root
        self.bench_filename = os.path.join(bench_root, "bench.npz")
        self.res_filename = os.path.join(bench_root, "order_count.npy")
        
        benchfile_shape, all_vals = run_benchmark(
            filename = self.bench_filename  ,
            return_array_descriptor = True  ,
            StopOnExcept = True             ,
        )
            
        compare_intent_order = {} # Recreate dict so that order is consistent with benchfile_shape

        for key, val in benchfile_shape.items():
            
            intent_in = compare_intent.get(key)
            
            if intent_in is None:
                
                if key in [repeat_ax_name, out_ax_name]:
                    intent = 'group'
                else:
                    intent = 'compare'
            
            else:
                intent = intent_in
            
            compare_intent_order[key] = intent

        self.compare_intent = compare_intent_order
        
        self.idx_all_group = []
        self.idx_all_compare = []

        self.name_group = []
        self.name_compare = []

        for i, (name, value) in enumerate(compare_intent.items()):
            
            if name == "group":
                self.idx_all_group.append(i)
                self.name_group.append(name)
            elif name == "compare":
                self.idx_all_compare.append(i)
                self.name_compare.append(name)
            
        self.n_group    = _prod_rel_shapes(self.idx_all_group     , self.all_vals.shape)
        self.n_compare  = _prod_rel_shapes(self.idx_all_compare   , self.all_vals.shape)
            
        self.k = k

        ncomb = math.comb(self.n_compare, self.k)
        nfac = math.factorial(self.k)

        if os.path.isfile(self.res_filename):
            
            order_count = np.load(self.res_filename)
            
            assert order_count.shape[0] == ncomb
            assert order_count.shape[1] == nfac
            
        else:
            order_count = np.zeros((ncomb, nfac), dtype=np.intp)

            