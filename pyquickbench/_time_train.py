import numpy as np
import time
import inspect
import warnings

from pyquickbench._defaults import *
from pyquickbench._utils import all_reductions

class TimeTrain():
    
    def __init__(
        self                    ,
        include_locs = True     ,
        name = ''               ,
        align_toc_names = False ,
        names_reduction = None  ,
    ):
        
        self.n = 0
        self.name = name
        self.all_tocs = [time.perf_counter()]
        self.all_tocs_record_time = [0.]
        self.all_tocs_record_time_sum = 0.
        self.all_tocs_names = []
        self.align_toc_names = align_toc_names
        self.max_name_len = 0
        self.include_locs = include_locs
        if self.include_locs:
            self.all_tocs_locs = []
            
        if names_reduction is None:
            self.names_reduction = None   
        else:
            self.names_reduction = all_reductions.get(names_reduction)
            if self.names_reduction is None:
                raise ValueError(f'Unknown reduction {names_reduction}')
        
        if self.include_locs and (self.names_reduction is not None):
            warnings.warn("include_locs and names_reduction were both set to True which results in reduced functionnality")
        
    def toc(self, name=''):
        
        tbeg = time.perf_counter()

        if name is None:
            name = str(self.n)
        elif not(isinstance(name, str)):
            name = str(name)
            
        self.n += 1
        self.all_tocs.append(tbeg)
        self.all_tocs_names.append(name)
        self.max_name_len = max(self.max_name_len, len(name))
        
        if self.include_locs:
            caller = inspect.getframeinfo(inspect.stack()[1][0])
            self.all_tocs_locs.append(f'{caller.filename}: {caller.lineno} in {caller.function}')
            
        tend = time.perf_counter()
        dt = tend-tbeg
        self.all_tocs_record_time.append(dt)
        self.all_tocs_record_time_sum += dt
        
    def get_recorded_time(self, idx):
        return self.all_tocs[idx+1]-(self.all_tocs[idx]+self.all_tocs_record_time[idx])
        
    def __repr__(self): 
        
        out = ''
        
        if self.name == '':
            out += f'TimeTrain results:\n\n'
        else:
            out += f'TimeTrain {self.name} results:\n\n'
            
        if self.names_reduction is None:
            
            for i in range(self.n):

                name = self.all_tocs_names[i]            
                
                if self.align_toc_names:
                    filler = ' ' * (self.max_name_len - len(name))
                else:
                    filler = ''
                
                if name != '':
                    out += f'{name}{filler}: '
                
                out += f'{self.get_recorded_time(i):.8f} s'
                if self.include_locs:
                    out += f' at {self.all_tocs_locs[i]}'
                    
                out += '\n'
        
        else:
            
            d = self.to_dict()
            for name, arr in d.items():
                
                if self.align_toc_names:
                    filler = ' ' * (self.max_name_len - len(name))
                else:
                    filler = ''
                
                if name != '':
                    out += f'{name}{filler}: '
                    
                out += f'{self.names_reduction(arr):.8f} s'
                if self.include_locs:
                    out += f' at {self.all_tocs_locs[i]}'
                    
                out += '\n'
        out += '\n'
        out += f'Total: {self.all_tocs[self.n]-self.all_tocs[0]-self.all_tocs_record_time_sum:.8f} s\n'
            
        return out
    
    def to_dict(self):
        
        dict_list = {}
        for i, name in enumerate(self.all_tocs_names):
            
            t = self.get_recorded_time(i)
            
            toclist = dict_list.get(name)
            if toclist is None:
                dict_list[name] = [t]
            else:
                toclist.append(t)
         
        res = {name:np.array(val) for name, val in dict_list.items()}
            
        return res
    