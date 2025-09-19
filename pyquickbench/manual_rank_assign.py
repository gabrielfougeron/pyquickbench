
import os
import math
import itertools
import collections.abc

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
    def restrict_values(self):
        
        res = {}
        
        for key, idx in zip(self.benchfile_names, self.restrict_idx):
            
            bench_vals = self.benchfile_shape[key]
            
            if hasattr(bench_vals, '__getitem__'):
            
                vals = []    
                for i in idx:
                    vals.append(bench_vals[i])
            
            else:
                vals = idx
            
            res[key] = vals
        
        return res
    
    @restrict_values.setter
    def restrict_values(self, res_val_in):
        self.restrict_idx, self.restrict_shape = self.complete_restrict_values(res_val_in)

    @property
    def compare_intent(self):
        return self._compare_intent
    
    @compare_intent.setter
    def compare_intent(self, d):
        self._compare_intent = self.complete_compare_intent(d)

        group_tuple, compare_tuple = self.divide_compare_intent(self.compare_intent)
        self.idx_all_group  , self.name_group  , self.n_group  , self.n_group_unres   = group_tuple
        self.idx_all_compare, self.name_compare, self.n_compare, self.n_compare_unres = compare_tuple

    def i_compare_res_to_unres(self, i_compare_res):

        idx_compare_res = _from_mem_shift_restricted(i_compare_res, self.idx_all_compare, self.restrict_shape)   

        idx_compare_unres = np.empty(len(self.restrict_shape), dtype=np.intp)        
        for ii, (j, jj) in enumerate(zip(idx_compare_res, self.idx_all_compare)):
            idx_compare_unres[ii] = self.restrict_idx[jj][j] 
                    
        return _mem_shift_restricted(idx_compare_unres, self.idx_all_compare ,self.all_vals.shape), idx_compare_unres
    
    def compute_iset_res_to_unres(self):
    
        iset_res_to_unres = np.empty(self.nset_res, dtype=np.intp)
        i_compare_arr_unres = np.empty(self.k, dtype=np.intp)
        idx_compare_unres = np.empty(len(self.restrict_shape), dtype=np.intp)
        
        for iset_res in range(self.nset_res):
            
            i_compare_arr_res = rankstats.unrank_combination(iset_res, self.n_compare, self.k)
            
            for i in range(self.k):
                
                idx_compare_res = _from_mem_shift_restricted(i_compare_arr_res[i], self.idx_all_compare, self.restrict_shape)   
                
                for ii, (j, jj) in enumerate(zip(idx_compare_res, self.idx_all_compare)):
                    idx_compare_unres[ii] = self.restrict_idx[jj][j] 

                i_compare_arr_unres[i] = _mem_shift_restricted(idx_compare_unres, self.idx_all_compare ,self.all_vals.shape)

            iset_res_to_unres[iset_res] = rankstats.rank_combination(i_compare_arr_unres, self.n_compare_unres, self.k)

        return iset_res_to_unres

    def __init__(
        self                    ,
        bench_root              ,
        compare_intent = {}     ,
        restrict_values = {}    ,
        k = 2                   ,
        iset_stiffness = 1.     ,
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
        
        self.benchfile_names = [key for key in self.benchfile_shape]

        self.iset_stiffness = iset_stiffness
        self.k = k
        self.restrict_values = restrict_values
        self.compare_intent = compare_intent
        
        self.nset_unres = math.comb(self.n_compare_unres, self.k)
        self.nset_res = math.comb(self.n_compare, self.k)
        
        self.iset_res_to_unres = self.compute_iset_res_to_unres()

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
    
    def complete_restrict_values(self, restrict_values = {}):
        
        restrict_idx = [] # Order is consistent with benchfile_shape

        for key, val in self.benchfile_shape.items():
            
            restrict_in = restrict_values.get(key)
            
            if restrict_in is None:
                
                if hasattr(val, "__len__"):
                    idx = range(len(val))
                else:
                    idx = range(val)

            else:
                
                idx = []
                
                for res_val in restrict_in:
                        
                    if isinstance(val, collections.abc.Iterable):
                        
                        for i, v in enumerate(val):
                            if v == res_val:
                                break
                        else:
                            i = None
                            
                    else:
                        
                        if (0 <= res_val) and (res_val < val):
                            i = res_val
                        else:
                            i = None
                    
                    if i is None:
                        raise ValueError(f"Could not find restricted value {res_val} in benchmark {key}:{val}.")
                    else:
                        idx.append(i)

            restrict_idx.append(idx)
            
        restrict_shape = np.array([len(idx) for idx in restrict_idx], dtype=np.intp)
        
        return restrict_idx, restrict_shape
    
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
             
        n_group    = _prod_rel_shapes(idx_all_group     , self.restrict_shape)
        n_compare  = _prod_rel_shapes(idx_all_compare   , self.restrict_shape)
        
        n_group_unres    = _prod_rel_shapes(idx_all_group     , self.all_vals.shape)
        n_compare_unres  = _prod_rel_shapes(idx_all_compare   , self.all_vals.shape)

        return (idx_all_group, name_group, n_group, n_group_unres), (idx_all_compare, name_compare, n_compare, n_compare_unres)

    def next_set(self, iset = None):
        
        if iset is None:
            iset = self.next_iset()
        
        i_group_arr = np.random.randint(self.n_group, size=self.k)
        i_compare_arr = rankstats.unrank_combination(iset, self.n_compare, self.k)

        idx_vals_arr = np.empty(self.all_vals.ndim, dtype=np.intp)

        vals_list = []
        for (i_group, i_compare) in zip(i_group_arr, i_compare_arr):
            
            idx_group   = _from_mem_shift_restricted(i_group  , self.idx_all_group  , self.restrict_shape)
            idx_compare = _from_mem_shift_restricted(i_compare, self.idx_all_compare, self.restrict_shape)

            for i, j in zip(idx_group, self.idx_all_group):
                idx_vals_arr[j] = self.restrict_idx[j][i]
            for i, j in zip(idx_compare, self.idx_all_compare):
                idx_vals_arr[j] = self.restrict_idx[j][i]

            idx_vals = tuple(idx_vals_arr)

            vals_list.append(self.all_vals[idx_vals])
        
        return iset, vals_list
    
    def next_iset(self):

        higher_means_likelier = (-self.iset_stiffness)*self.n_votes_set[self.iset_res_to_unres]
        p = scipy.special.softmax(higher_means_likelier)

        return np.random.choice(self.nset_res, p=p)
    
    def vote_for_ibest(self, iset_res, ibest, mul = 1):
        
        iset_unres = self.iset_res_to_unres[iset_res]
        
        if ibest < 0:
            # When you really don't know what to vote for. Since best_count has dtype np.intp, don't do halves.
            self.best_count[iset_unres, :] += mul
            self.n_votes_set[iset_unres] += self.best_count.shape[1]*mul
        else:
            self.best_count[iset_unres, ibest] += mul
            self.n_votes_set[iset_unres] += mul
    
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
            
            self.best_count = np.zeros((self.nset_unres, self.k), dtype=np.intp)
            loaded_compare_intent = {}
            
        self.compare_intent = self.complete_finer_compare_intent(loaded_compare_intent)
        
        if self.nset_unres != self.best_count.shape[0]:
            raise ValueError(f'Incompatible compare intents with loaded best_count')
        
        self.n_votes_set = self.best_count.sum(axis=1)
    
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
            best_count = np.ascontiguousarray(self.best_count[self.iset_res_to_unres,:])
            
        else:
            
            # raise NotImplementedError
            finer_compare_intent = self.complete_finer_compare_intent(compare_intent)

            _, (idx_all_compare, name_compare, n_compare, n_compare_unres) = self.divide_compare_intent(finer_compare_intent)

            i_compare_fused = [[] for ifuse in range(n_compare)]

            for self_idx_compare_res in itertools.product(*[range(self.restrict_shape[i]) for i in self.idx_all_compare]):
                        
                idx_compare = []
                for i_cmp_new in idx_all_compare:
                    i_cmp = self.idx_all_compare.index(i_cmp_new)
                    idx_compare.append(self_idx_compare_res[i_cmp])
                    
                self_i_compare = _mem_shift_restricted(self_idx_compare_res  , self.idx_all_compare  , self.restrict_shape)
                i_compare = _mem_shift_restricted(idx_compare  , idx_all_compare  , self.restrict_shape)
                                
                i_compare_fused[i_compare].append(self_i_compare)
                
            best_count = rankstats.fuse_score_to_partial_best_count(np.ascontiguousarray(self.best_count[self.iset_res_to_unres,:]), i_compare_fused)


        n_votes = best_count.sum()

        A = rankstats.build_sinkhorn_mat(n_compare, self.k)
        p, q = rankstats.build_sinkhorn_rhs_new(best_count, reg_eps = 0.00000)
        
        reg_beta = 0.
        reg_alpham1 = 0.
        u, v = sinkhorn_knopp(A, p, q, reg_alpham1 = reg_alpham1, reg_beta = reg_beta)

        compare_order = np.argsort(v)
        
        v_scal = v / v.sum()
        
        # logu, logv = rankstats.uv_to_loguv(u, v)
        
        compare_args = []
        for i_compare_res in compare_order:
            
            idx_compare_res = _from_mem_shift_restricted(i_compare_res, idx_all_compare, self.restrict_shape)   

            idx_compare_unres = np.empty(len(self.restrict_shape), dtype=np.intp)        
            for ii, (j, jj) in enumerate(zip(idx_compare_res, idx_all_compare)):
                idx_compare_unres[ii] = self.restrict_idx[jj][j] 
            
            args = {}
            
            for key, idx in zip(name_compare, idx_compare_unres):

                val = self.benchfile_shape.get(key)
                
                if hasattr(val, '__getitem__'):
                    args[key] = val[idx]
                else:
                    args[key] = idx

            compare_args.append(args)

        return n_votes, compare_args, v_scal[compare_order]
            