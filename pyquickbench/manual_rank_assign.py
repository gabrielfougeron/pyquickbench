
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

all_vote_modes = ["best", "order"]

class ManualRankAssign():
    
    @property
    def restrict_values(self):
        return self.build_restrict_values(self.benchfile_shape, self.restrict_idx)

    @staticmethod
    def build_restrict_values(benchfile_shape, restrict_idx):
        
        res = {}
        
        for (key, bench_vals), idx in zip(benchfile_shape.items(), restrict_idx):
            
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

        group_tuple, compare_tuple = self.divide_compare_intent(self._compare_intent)
        self.idx_all_group  , self.name_group  , self.n_group  , self.n_group_unres   = group_tuple
        self.idx_all_compare, self.name_compare, self.n_compare, self.n_compare_unres = compare_tuple
        
        self.nset_unres = math.comb(self.n_compare_unres, self.k)
        self.nset_res = math.comb(self.n_compare, self.k)
        
        self.iset_res_to_unres = self.compute_iset_res_to_unres()

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
        *                       ,
        bench_root = './'       ,
        benchfile_shape = None  ,
        all_vals = None         ,
        compare_intent = {}     ,
        restrict_values = {}    ,
        vote_count = None       ,
        vote_mode = None        ,
        k = 2                   ,
        iset_stiffness = 1.     ,
        
    ):
        
        self.cur_icompare = -1
        
        self.bench_root = bench_root
        self.benchfile_shape = benchfile_shape
        self.all_vals = all_vals
        
        self.benchfile_names = [key for key in self.benchfile_shape]

        self.iset_stiffness = iset_stiffness
        self.k = k
        if vote_mode is None:
            self.vote_mode = "best"
        else:
            if vote_mode in all_vote_modes:
                self.vote_mode = vote_mode
            else:
                raise ValueError(f"Unknown {vote_mode = }. Possible values are {all_vote_modes}.")
        
        if self.vote_mode == "best":
            self.nopt_per_set = self.k
            
        elif self.vote_mode == "order":
            self.nopt_per_set = math.factorial(self.k)
            
        self.restrict_values = restrict_values
        self.compare_intent = compare_intent
        
        if vote_count is not None:
            if vote_count.shape != (self.nset_unres , self.nopt_per_set):
                vote_count = None
                # raise ValueError(f"Received vote_count with wrong shape. Expected {(self.nset_unres , self.nopt_per_set)}, received {vote_count.shape}.")
            
        if vote_count is None:
            vote_count = np.zeros((self.nset_unres, self.nopt_per_set), dtype=np.intp)
            
        self.vote_count = vote_count
        self.n_votes_set = self.vote_count.sum(axis=1)

    def get_img_path(self, val):
        
        img_path = os.path.join(self.bench_root, "imgs", f"image_{str(int(val)).zfill(5)}_.png")
        
        return img_path

    def print_restrict_bench(self):
        
        restrict_values = self.restrict_values
        
        for key, vals in restrict_values.items():

            print(key, ":")
            
            if hasattr(vals, '__getitem__'):
                for val in vals:
                    print("    ", val)
            elif hasattr(vals, '__len__'):
                i = len(vals)-1
                if i > 0:
                    print("    ",f"0 - {i}")
                else:
                    print("    ",f"{i}")
            else:
                i = vals-1
                if i > 0:
                    print("    ",f"0 - {i}")
                else:
                    print("    ",f"{i}")
                    
            print()

    @staticmethod
    def default_compare_intent(benchfile_shape, compare_intent = {}):
        
        compare_intent_out = {} # Recreate dict no matter what so that order is consistent with benchfile_shape

        for key, val in benchfile_shape.items():
            
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

    def complete_compare_intent(self, compare_intent = {}):
        return self.default_compare_intent(self.benchfile_shape, compare_intent = compare_intent)
    
    @staticmethod
    def default_restrict_values(benchfile_shape, restrict_values = {}):
        
        restrict_idx = [] # Order is consistent with benchfile_shape

        for key, val in benchfile_shape.items():
            
            restrict_in = restrict_values.get(key)
            
            if restrict_in is None:
                
                if hasattr(val, "__len__"):
                    idx = list(range(len(val)))
                else:
                    idx = list(range(val))

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
    
    def complete_restrict_values(self, restrict_values = {}):
        return self.default_restrict_values(self.benchfile_shape, restrict_values = restrict_values)
    
    def complete_finer_compare_intent(self, fine_compare_intent):
        
        compare_intent_out = {} # Recreate dict no matter what so that order is consistent with compare_intent

        for key, init_intent in self._compare_intent.items():
            
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

    def fuse_compare_intent(self, compare_intent = None):

        if compare_intent is None:
            
            idx_all_compare = self.idx_all_compare
            name_compare = self.name_compare
            n_compare = self.n_compare
            vote_count = np.ascontiguousarray(self.vote_count[self.iset_res_to_unres,:])
            
        else:
            
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

            vote_count = rankstats.fuse_score_to_partial_vote_count(self.vote_mode, np.ascontiguousarray(self.vote_count[self.iset_res_to_unres,:]), i_compare_fused)

        return idx_all_compare, name_compare, n_compare, vote_count

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
            # When you really don't know what to vote for. Since vote_count has dtype np.intp, don't do halves.
            self.vote_count[iset_unres, :] += mul
            self.n_votes_set[iset_unres] += self.vote_count.shape[1]*mul
        else:
            self.vote_count[iset_unres, ibest] += mul
            self.n_votes_set[iset_unres] += mul
    
    def save_results(self, vote_count_filename = None):
        
        if vote_count_filename is None:
            vote_count_filename = self.vote_count_filename
        
        file_bas, file_ext = os.path.splitext(vote_count_filename)
            
        if file_ext == '.npy':
            np.save(vote_count_filename, self.vote_count)   
            
        elif file_ext == '.npz':

            np.savez(
                vote_count_filename                     ,
                vote_count = self.vote_count            ,
                vote_mode = self.vote_mode              ,
                compare_intent = self.compare_intent    ,
                all_vals = self.all_vals                ,
                **self.benchfile_shape                  ,
            )
            
        else:
            raise ValueError(f'Unknown file extension {file_ext}')
        
    def get_order(self, compare_intent = None):
        
        idx_all_compare, name_compare, n_compare, vote_count = self.fuse_compare_intent(compare_intent = compare_intent)
        
        if (n_compare < self.k):
            raise ValueError("Not enough items to compare")
        
        n_votes = vote_count.sum()
        
        if self.vote_mode != "best":
            raise NotImplementedError
                
        if (n_votes < 1):
            
            nvec, k = rankstats.find_nvec_k(vote_count.shape[0], vote_count.shape[1], vote_mode = self.vote_mode)
            v = np.ones(nvec, dtype=np.float64)
            
        else:

            A = rankstats.build_sinkhorn_best_count_mat(n_compare, self.k)
            reg_eps = 1./(n_votes+1)
            p, q = rankstats.build_sinkhorn_rhs_new(vote_count, reg_eps = reg_eps)

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
            