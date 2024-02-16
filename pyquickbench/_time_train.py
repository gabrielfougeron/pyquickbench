import numpy as np
import time
import inspect

from pyquickbench._defaults import *

class TimeTrain():
    
    def __init__(
        self                    ,
        include_locs = True     ,
        name=''                 ,
        align_toc_names = False ,
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
        
    # def to_dict(self):
    #     return {name: toc for (name, toc) in zip(self.all_tocs_names, self.all_tocs)}
    
    def __repr__(self): 
        
        out = ''
        
        if self.name == '':
            out += f'TimeTrain results:\n\n'
        else:
            out += f'TimeTrain {self.name} results:\n\n'
            
        
        for i in range(self.n):

            name = self.all_tocs_names[i]            
            
            if self.align_toc_names:
                filler = ' ' * (self.max_name_len - len(name))
            else:
                filler = ''
            
            if name != '':
                out += f'{name}{filler}: '
            
            out += f'{self.all_tocs[i+1]-(self.all_tocs[i]+self.all_tocs_record_time[i]):.8f} s'
            if self.include_locs:
                out += f' at {self.all_tocs_locs[i]}'
                
            out += '\n'
            
        out += '\n'
        out += f'Total: {self.all_tocs[self.n]-self.all_tocs[0]-self.all_tocs_record_time_sum:.8f} s\n'
            
        return out