
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



import logging
# logging.basicConfig(filename='myapp.log', level=logging.INFO)
logging.basicConfig(
    filename='myapp.log',
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger(__name__)




all_vote_modes = ["best", "order"]
all_store_modes = ["best", "order"]

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
        
        self.nset_store_unres = math.comb(self.n_compare_unres, self.nchoices_store)
        self.nset_store_res = math.comb(self.n_compare, self.nchoices_store)

        assert self.nset_store_unres <= np.iinfo(np.intp).max
        assert self.nset_store_res <= np.iinfo(np.intp).max
        
        self.iset_res_to_unres = self.compute_iset_res_to_unres()

    def i_compare_res_to_unres(self, i_compare_res):

        idx_compare_res = _from_mem_shift_restricted(i_compare_res, self.idx_all_compare, self.restrict_shape)   

        idx_compare_unres = np.empty(len(self.restrict_shape), dtype=np.intp)        
        for ii, (j, jj) in enumerate(zip(idx_compare_res, self.idx_all_compare)):
            idx_compare_unres[ii] = self.restrict_idx[jj][j] 
                    
        return _mem_shift_restricted(idx_compare_unres, self.idx_all_compare ,self.all_vals.shape), idx_compare_unres
    
    def compute_iset_res_to_unres(self):
    
        iset_res_to_unres = np.empty(self.nset_store_res, dtype=np.intp)
        i_compare_arr_unres = np.empty(self.nchoices_store, dtype=np.intp)
        idx_compare_unres = np.empty(len(self.restrict_shape), dtype=np.intp)
        
        for iset_res in range(self.nset_store_res):
            
            i_compare_arr_res = rankstats.unrank_combination(iset_res, self.n_compare, self.nchoices_store)
            
            for i in range(self.nchoices_store):
                
                idx_compare_res = _from_mem_shift_restricted(i_compare_arr_res[i], self.idx_all_compare, self.restrict_shape)   
                
                for ii, (j, jj) in enumerate(zip(idx_compare_res, self.idx_all_compare)):
                    idx_compare_unres[ii] = self.restrict_idx[jj][j] 

                i_compare_arr_unres[i] = _mem_shift_restricted(idx_compare_unres, self.idx_all_compare ,self.all_vals.shape)

            iset_res_to_unres[iset_res] = rankstats.rank_combination(i_compare_arr_unres, self.n_compare_unres, self.nchoices_store)

        return iset_res_to_unres

    def __init__(
        self                    ,
        *                       ,
        bench_root = './'       ,
        benchfile_shape = None  ,
        all_vals = None         ,
        compare_intent = {}     ,
        restrict_values = {}    ,
        vote_mode = None        ,
        nchoices_vote = 2       ,
        store_mode = None       ,
        nchoices_store = 2      ,
        store_count = None      ,
        iset_stiffness = 1.     ,
        
    ):
        
        self.cur_icompare = -1
        
        self.bench_root = bench_root
        self.benchfile_shape = benchfile_shape
        self.all_vals = all_vals
        
        self.benchfile_names = [key for key in self.benchfile_shape]

        self.iset_stiffness = iset_stiffness
        
        if nchoices_vote < nchoices_store:
            raise ValueError(f"nchoices_vote should be greater or equal than nchoices_store. Received {nchoices_vote = } < {nchoices_store = }.")

        self.nchoices_vote = nchoices_vote
        self.nchoices_store = nchoices_store

        if vote_mode is None:
            self.vote_mode = "best"
        else:
            if vote_mode in all_vote_modes:
                self.vote_mode = vote_mode
            else:
                raise ValueError(f"Unknown {vote_mode = }. Possible values are {all_vote_modes}.")

        if store_mode is None:
            self.store_mode = "best"
        else:
            if store_mode in all_store_modes:
                self.store_mode = store_mode
            else:
                raise ValueError(f"Unknown {store_mode = }. Possible values are {all_store_modes}.")

        if self.vote_mode == "best":
            self.nopts_vote = self.nchoices_vote
            
        elif self.vote_mode == "order":
            self.nopts_vote = math.factorial(self.nchoices_vote)
            
        if self.store_mode == "best":
            self.nopts_store = self.nchoices_store
            
        elif self.store_mode == "order":
            self.nopts_store = math.factorial(self.nchoices_store)

        # TODO Further restrict incompatibilities between vote_mode and store_mode ?

        # Lots is happening here. Those are properties with custom setters.
        self.restrict_values = restrict_values
        self.compare_intent = compare_intent
        
        
        logger.info("init MRA")
        logger.info(self.compare_intent)
        

        assert math.comb(self.n_compare, self.nchoices_vote) <= np.iinfo(np.intp).max    
        
        if store_count is not None:
            if store_count.shape != (self.nset_store_unres , self.nopts_store):
                # store_count = None
                raise ValueError(f"Received store_count with wrong shape. Expected {(self.nset_store_unres , self.nopts_store)}, received {store_count.shape}.")
            
        if store_count is None:
            store_count = np.zeros((self.nset_store_unres, self.nopts_store), dtype=np.intp)
            
        self.store_count = store_count
        self.n_store_set = self.store_count.sum(axis=1)

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
                            
                            is_eq = v == res_val

                            if hasattr(is_eq, 'all'):
                                is_eq = is_eq.all()
                                
                            if is_eq:
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
            store_count = np.ascontiguousarray(self.store_count[self.iset_res_to_unres,:])
            
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

            store_count = rankstats.fuse_score_to_partial_vote_count(self.store_mode, np.ascontiguousarray(self.store_count[self.iset_res_to_unres,:]), i_compare_fused)

        return idx_all_compare, name_compare, n_compare, store_count

    def next_set(self, iset_res = None):

        if iset_res is None:
            iset_res = self.next_iset_res()

        # i_group_arr = np.random.randint(self.n_group, size=self.nchoices_vote)
        i_group_arr = [np.random.randint(self.n_group)]*self.nchoices_vote
        
        i_compare_arr = rankstats.unrank_combination(iset_res, self.n_compare, self.nchoices_vote)

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

        return iset_res, vals_list
    
    def next_iset_res(self):
        
        # Probability measure on restricted stored sets
        p = scipy.special.softmax((-self.iset_stiffness)*self.n_store_set[self.iset_res_to_unres])

        chosen_elems_set = set()
        
        while len(chosen_elems_set) < self.nchoices_vote:
            
            iset_res = np.random.choice(self.nset_store_res, p=p)
            elems_res = rankstats.unrank_combination(iset_res, self.n_compare, self.nchoices_store)

            for elem in elems_res:
                
                chosen_elems_set.add(elem)
                
                if len(chosen_elems_set) == self.nchoices_vote:
                    break

        chosen_elems_arr = np.sort(np.fromiter(chosen_elems_set, dtype=np.intp))

        iset_res = rankstats.rank_combination(chosen_elems_arr, self.n_compare, self.nchoices_vote)
        
        assert 0 <= iset_res
        assert iset_res <= math.comb(self.n_compare, self.nchoices_vote)
        
        return iset_res
    
    def vote_for_ibest(self, iset_vote_res, ibest_vote, mul = 1):

        if ibest_vote < 0:
            ibest_vote_list = list(range(self.nopts_vote))
        else:
            ibest_vote_list = [ibest_vote]

        if self.vote_mode == "order" and self.store_mode == "order":
            vote_to_store_reduce = rankstats.single_order_to_lower_k
        elif self.vote_mode == "order" and self.store_mode == "best":
            vote_to_store_reduce = rankstats.single_order_to_best_lower_k
        elif self.vote_mode == "best" and self.store_mode == "best":
            vote_to_store_reduce = rankstats.single_best_to_lower_k
        else:
            raise NotImplementedError
        
        for ibest in ibest_vote_list:

            iset_store_res_list, ibest_store_list = vote_to_store_reduce(iset_vote_res, ibest, self.n_compare, self.nchoices_vote, self.nchoices_store)

            for iset_store_res, ibest_store in zip(iset_store_res_list, ibest_store_list):

                iset_store_unres = self.iset_res_to_unres[iset_store_res]
                self.store_count[iset_store_unres, ibest_store] += mul
                self.n_store_set[iset_store_unres] += mul

    def save_results(self, store_count_filename = None):
        
        if store_count_filename is None:
            store_count_filename = self.store_count_filename
        
        file_bas, file_ext = os.path.splitext(store_count_filename)
            
        if file_ext == '.npy':
            np.save(store_count_filename, self.store_count)   
            
        elif file_ext == '.npz':

            np.savez(
                store_count_filename                    ,
                store_count = self.store_count          ,
                store_mode = self.store_mode            ,
                compare_intent = self.compare_intent    ,
                all_vals = self.all_vals                ,
                **self.benchfile_shape                  ,
            )
            
        else:
            raise ValueError(f'Unknown file extension {file_ext}')
        
    def get_order(self, compare_intent = None):
        
        idx_all_compare, name_compare, n_compare, store_count = self.fuse_compare_intent(compare_intent = compare_intent)
        
        if (n_compare < self.nchoices_store):
            raise ValueError("Not enough items to compare")
        
        n_store = store_count.sum()
        
        if self.store_mode != "best":
            raise NotImplementedError
                
        if (n_store < 1):
            
            nvec, k = rankstats.find_nvec_k(store_count.shape[0], store_count.shape[1], vote_mode = self.store_mode)
            v = np.ones(nvec, dtype=np.float64)
            
        else:
            
            reg_eps = 1./(n_store+1)
            A, p, q = rankstats.build_sinkhorn_problem(store_count, self.store_mode, reg_eps = reg_eps)

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

        return n_store, compare_args, v_scal[compare_order]
            