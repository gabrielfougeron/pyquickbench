
import os
import math
import numpy as np
import scipy

from . import rankstats

from pyquickbench._benchmark import run_benchmark
from pyquickbench._defaults import *
from pyquickbench._utils import _prod_rel_shapes

class ManualRankAssign():

    def __init__(
        self                    ,
        bench_root              ,
        compare_intent  = {}    ,
        k = 2                   ,
    ):
        
        self.cur_icompare = -1
        
        self.bench_root = bench_root
        self.bench_filename = os.path.join(bench_root, "bench.npz")
        self.best_count_filename = os.path.join(bench_root, "best_count.npy")
        
        self.benchfile_shape, self.all_vals = run_benchmark(
            filename = self.bench_filename  ,
            return_array_descriptor = True  ,
            StopOnExcept = True             ,
        )
            
        compare_intent_order = {} # Recreate dict so that order is consistent with benchfile_shape

        for key, val in self.benchfile_shape.items():
            
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

        for i, (name, value) in enumerate(self.compare_intent.items()):
            
            if value == "group":
                self.idx_all_group.append(i)
                self.name_group.append(name)
                
            elif value == "compare":
                self.idx_all_compare.append(i)
                self.name_compare.append(name)
            else:
                raise ValueError(f'Unknown compare intent {value}')
             
        self.n_group    = _prod_rel_shapes(self.idx_all_group     , self.all_vals.shape)
        self.n_compare  = _prod_rel_shapes(self.idx_all_compare   , self.all_vals.shape)

        self.k = k
        self.ncomb = math.comb(self.n_compare, self.k)

        if os.path.isfile(self.best_count_filename):
            
            self.best_count = np.load(self.best_count_filename)
            
            if (self.best_count.shape[0] != self.ncomb) or (self.best_count.shape[1] != self.k):
                
                raise ValueError(f'Best count in file {self.best_count_filename} has wrong shape. Expected {(self.ncomb, self.k)}, received {self.best_count.shape}.')
            
        else:
            
            self.best_count = np.zeros((self.ncomb, self.k), dtype=np.intp)

    def next_set(self):
        
        self.next_iset()
        i_group_arr = np.random.randint(self.n_group)
        i_compare_arr = rankstats.unrank_combination(self.cur_iset, self.n_compare, self.k)

        # vals_list = []
        # for (i_group, i_compare) in zip(i_group_arr, i_compare_arr):
            
            # i_color, idx_curve_color = _get_rel_idx_from_maze(idx_all_curve_color, idx_vals, all_vals.shape)
            
        
        # return
    
    def next_iset(self):
        self.cur_iset = np.argmin(self.best_count.sum(axis=1))
    
    def vote_for_ibest(self, ibest):

        self.best_count[self.cur_iset, ibest] += 1
    
    def save_results(self):
        
        np.save(self.best_count_filename, self.best_count)