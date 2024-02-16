import numpy as np
import time
import inspect

from pyquickbench._defaults import *

class TimeTrain():
    
    def __init__(self, IncludeLocs=True, name=''):
        
        self.n = 0
        self.name = name
        self.all_tocs = [time.perf_counter()]
        self.all_tocs_names = []
        self.IncludeLocs = IncludeLocs
        if self.IncludeLocs:
            self.all_tocs_locs = []
        
    def toc(self, name=''):

        if name is None:
            name = str(self.n)
        elif not(isinstance(name, str)):
            name = str(name)
            
        self.n += 1
        self.all_tocs.append(time.perf_counter())
        self.all_tocs_names.append(name)
        
        if self.IncludeLocs:
            caller = inspect.getframeinfo(inspect.stack()[1][0])
            self.all_tocs_locs.append(f'{caller.filename}: {caller.lineno}')

            
    # def to_dict(self):
    #     return {name: toc for (name, toc) in zip(self.all_tocs_names, self.all_tocs)}
    
    def __repr__(self):
        
        out = f'TimeTrain {self.name} results:\n\n'
        
        for i in range(self.n):
            
            name = self.all_tocs_names[i]
            if name != '':
                out += f'{name}: '
            
            out += f'{self.all_tocs[i+1]-self.all_tocs[i]}'
            if self.IncludeLocs:
                out += f' at {self.all_tocs_locs[i]}'
                
            out += '\n'
            
        out += '\n'
        out += f'Total: {self.all_tocs[self.n]-self.all_tocs[0]}\n'
            
        return out