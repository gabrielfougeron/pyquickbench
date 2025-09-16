
import os
import math
import itertools

import numpy as np
import scipy

from . import rankstats

from pyquickbench.cython.sinkhorn import sinkhorn_knopp

from pyquickbench._benchmark import run_benchmark
from pyquickbench._defaults import *
from pyquickbench._utils import (
    _prod_rel_shapes            ,
    _mem_shift_restricted       ,
    _from_mem_shift_restricted  ,
    _get_rel_idx_from_maze      ,
)

class ManualRankAssign():
    
    @property
    def compare_intent(self):
        return self._compare_intent
    
    @compare_intent.setter
    def compare_intent(self, d):
        self._compare_intent = self.complete_compare_intent(d)

        group_tuple, compare_tuple = self.divide_compare_intent(self.compare_intent)
        self.idx_all_group  , self.name_group  , self.n_group = group_tuple
        self.idx_all_compare, self.name_compare, self.n_compare = compare_tuple
        
        self.ncomb = math.comb(self.n_compare, self.k)

    def __init__(
        self                ,
        bench_root          ,
        compare_intent = {} ,
        k = 2               ,
    ):
        
        self.cur_icompare = -1
        
        self.bench_root = bench_root
        self.bench_filename = os.path.join(self.bench_root, "bench.npz")
        self.best_count_filename = os.path.join(self.bench_root, "best_count.npz")
        
        self.benchfile_shape, self.all_vals = run_benchmark(
            filename = self.bench_filename  ,
            return_array_descriptor = True  ,
            StopOnExcept = True             ,
        )

        self.k = k

        self.compare_intent = compare_intent
        
        self.load_results()

    def complete_compare_intent(self, compare_intent = {}):
        
        compare_intent_out = {} # Recreate dict no matter what so that order is consistent with benchfile_shape

        for key, val in self.benchfile_shape.items():
            
            intent_in = compare_intent.get(key)
            
            if intent_in is None:
                
                if key in [repeat_ax_name, out_ax_name]:
                    intent = 'group'
                else:
                    intent = 'compare'
            
            else:
                intent = intent_in
            
            compare_intent_out[key] = intent
        
        return compare_intent_out
    
    def complete_finer_compare_intent(self, fine_compare_intent):
        
        compare_intent_out = {} # Recreate dict no matter what so that order is consistent with compare_intent

        for key, init_intent in self.compare_intent.items():
            
            fine_intent = fine_compare_intent.get(key, init_intent)
            
            if (fine_intent == 'compare') and (init_intent == 'group'):
                raise ValueError(f"Cannot compare {key} than was previously grouped")

            compare_intent_out[key] = fine_intent
        
        return compare_intent_out
        
    def divide_compare_intent(self, compare_intent):
        
        idx_all_group = []
        idx_all_compare = []

        name_group = []
        name_compare = []

        for i, (name, value) in enumerate(compare_intent.items()):
            
            if value == "group":
                idx_all_group.append(i)
                name_group.append(name)
                
            elif value == "compare":
                idx_all_compare.append(i)
                name_compare.append(name)
            else:
                raise ValueError(f'Unknown compare intent {value}')
             
        n_group    = _prod_rel_shapes(idx_all_group     , self.all_vals.shape)
        n_compare  = _prod_rel_shapes(idx_all_compare   , self.all_vals.shape)

        return (idx_all_group, name_group, n_group), (idx_all_compare, name_compare, n_compare)

    def next_set(self):
        
        iset = self.next_iset()
        
        i_group_arr = np.random.randint(self.n_group, size=self.k)
        i_compare_arr = rankstats.unrank_combination(iset, self.n_compare, self.k)

        idx_vals_arr = np.empty(self.all_vals.ndim, dtype=np.intp)

        vals_list = []
        for (i_group, i_compare) in zip(i_group_arr, i_compare_arr):
            
            idx_group   = _from_mem_shift_restricted(i_group  , self.idx_all_group  , self.all_vals.shape)
            idx_compare = _from_mem_shift_restricted(i_compare, self.idx_all_compare, self.all_vals.shape)

            for i, j in zip(idx_group, self.idx_all_group):
                idx_vals_arr[j] = i
            for i, j in zip(idx_compare, self.idx_all_compare):
                idx_vals_arr[j] = i

            idx_vals = tuple(idx_vals_arr)

            vals_list.append(self.all_vals[idx_vals])
        
        return iset, vals_list
    
    def next_iset(self):
        
        # Deterministic arg min
        # return np.argmin(self.best_count.sum(axis=1))
        
        # "Soft" random arg min
        return np.random.choice(self.ncomb, p=scipy.special.softmax(-self.best_count.sum(axis=1)))
    
    def vote_for_ibest(self,cur_iset, ibest):
        
        if ibest < 0:
            # When you really don't know what to vote for. Since best_count has dtype np.intp, don't do halves.
            self.best_count[cur_iset, :] += 1
        else:
            self.best_count[cur_iset, ibest] += 1
    
    def load_results(self):
        
        file_bas, file_ext = os.path.splitext(self.best_count_filename)
        
        if os.path.isfile(self.best_count_filename):
            
            if file_ext == '.npy':
                self.best_count = np.load(self.best_count_filename) 
                loaded_compare_intent = {}
                
            elif file_ext == '.npz':
                
                file_content = np.load(self.best_count_filename)
                self.best_count = file_content['best_count']
                
                loaded_compare_intent = {key:val for (key,val) in file_content.items() if key!='best_count'}

            else:
                raise ValueError(f'Unknown file extension {file_ext}')

        else:
            
            self.best_count = np.zeros((self.ncomb, self.k), dtype=np.intp)
            loaded_compare_intent = {}
            
        self.compare_intent = self.complete_finer_compare_intent(loaded_compare_intent)
        
        if self.ncomb != self.best_count.shape[0]:
            raise ValueError(f'Incompatible compare intents with loaded best_count')
    
    def save_results(self):
        
        # self.best_count_filename = os.path.join(self.bench_root, "best_count.npz")

        file_bas, file_ext = os.path.splitext(self.best_count_filename)
            
        if file_ext == '.npy':
            np.save(self.best_count_filename, self.best_count)   
            
        elif file_ext == '.npz':

            np.savez(
                self.best_count_filename        ,
                best_count = self.best_count    ,
                **self.compare_intent           ,
            )
            
        else:
            raise ValueError(f'Unknown file extension {file_ext}')
        
    def get_order(self, compare_intent = None):
        
        if compare_intent is None:
            
            idx_all_compare = self.idx_all_compare
            name_compare = self.name_compare
            n_compare = self.n_compare
            best_count = self.best_count
            
        else:
            finer_compare_intent = self.complete_finer_compare_intent(compare_intent)

            (idx_all_group, name_group, n_group), (idx_all_compare, name_compare, n_compare) = self.divide_compare_intent(finer_compare_intent)

            i_compare_fused = [[] for ifuse in range(n_compare)]

            for self_idx_compare in itertools.product(*[range(self.all_vals.shape[i]) for i in self.idx_all_compare]):
                        
                idx_compare = []
                for i_cmp_new in idx_all_compare:
                    i_cmp = self.idx_all_compare.index(i_cmp_new)
                    idx_compare.append(self_idx_compare[i_cmp])
                    
                self_i_compare = _mem_shift_restricted(self_idx_compare  , self.idx_all_compare  , self.all_vals.shape)
                i_compare = _mem_shift_restricted(idx_compare  , idx_all_compare  , self.all_vals.shape)
                                
                i_compare_fused[i_compare].append(self_i_compare)
                
            best_count = rankstats.fuse_score_to_partial_best_count(self.best_count, i_compare_fused)

        A = rankstats.build_sinkhorn_mat(n_compare, self.k)
        p, q = rankstats.build_sinkhorn_rhs_new(best_count, reg_eps = 0.00000)
        
        reg_beta = 0.
        reg_alpham1 = 0.
        u, v = sinkhorn_knopp(A, p, q, reg_alpham1 = reg_alpham1, reg_beta = reg_beta)

        compare_order = np.argsort(v)
        
        v_scal = v / v.sum()
        
        # logu, logv = rankstats.uv_to_loguv(u, v)
        
        compare_args = []
        for i_compare in compare_order:
            
            idx_compare = _from_mem_shift_restricted(i_compare, idx_all_compare, self.all_vals.shape)
            
            args = {}
            
            for key, idx in zip(name_compare, idx_compare):

                val = self.benchfile_shape.get(key)
                
                if hasattr(val, '__getitem__'):
                    args[key] = val[idx]
                else:
                    args[key] = idx

            compare_args.append(args)

        return compare_args, v_scal[compare_order]
            