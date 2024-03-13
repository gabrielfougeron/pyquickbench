import os
import numpy as np
import time
import inspect
import warnings
import typing
import numpy.typing

from pyquickbench._defaults import *

class TimeTrain():
    """ Records elapsed time between interest points in code"""
    
    def __init__(
        self                                                    ,
        path_prefix         : typing.Union[str , None] = None   ,
        include_locs        : typing.Union[bool, None] = None   ,
        include_filename    : bool = True                       ,
        include_lineno      : bool = True                       ,
        include_funname     : bool = True                       ,
        name                : str  = ''                         ,
        align_toc_names     : bool = True                      ,
        names_reduction     : typing.Union[str , None] = None   ,
    ):
        """ Returns a TimeTrain

        Parameters
        ----------
        path_prefix : str | None, optional
            Path relative to which other paths are to be understood, by default None
        include_locs : bool | None, optional
            Whether to include locations in code when printing the TimeTrain, by default None
        include_filename : bool, optional
            Whether to include the file name in locations in code when printing the TimeTrain, by default True
        include_lineno : bool, optional
            Whether to include the line number in locations in code when printing the TimeTrain, by default True
        include_funname : bool, optional
            Whether to include the function name in locations in code when printing the TimeTrain, by default True
        name : str, optional
            Name of the TimeTrain, by default ''
        align_toc_names : bool, optional
            Whether to align toc names when printing the TimeTrain, by default True
        names_reduction : str | None, optional
            Reduction to be applied to tocs that share the same name, by default None

        """    
        
        self.n = 0
        self.name = name
        self.all_tocs_record_time = [0.]
        self.all_tocs_record_time_sum = 0.
        self.all_tocs_names = []
        self.align_toc_names = align_toc_names
        self.max_name_len = 0
        self.name_set = set()
        self.n_names = 0
        
        if include_locs is None:
            include_locs = include_filename or include_lineno or include_funname
            
        if path_prefix is None:
            path_prefix = os.path.abspath(os.path.join(inspect.stack()[1][1], os.pardir))

        self.path_prefix = path_prefix
        
        self.include_locs = include_locs
        
        self.include_filename = include_filename
        self.include_funname = include_funname
        self.include_lineno = include_lineno
        
        if self.include_locs:
            self.all_tocs_locs = []
            
        if names_reduction is None:
            self.names_reduction = None   
        else:
            self.names_reduction = all_reductions.get(names_reduction)
            if self.names_reduction is None:
                raise ValueError(f'Unknown reduction {names_reduction}')
        
        if self.include_locs and (self.names_reduction is not None):
            warnings.warn("include_locs and names_reduction were both enabled. Only the first location will be displayed for every name.")
        
        # This line goes at the very end for more precise measurements
        self.all_tocs = [time.perf_counter()]
        
        
    def toc(
        self                ,
        name    : str = ''  ,
    ):
        """
        Records a new wagon in the TimeTrain

        Parameters
        ----------
        name : str, optional
            _description_, by default ''
        """        
        
        tbeg = time.perf_counter()

        if name is None:
            name = str(self.n)
        elif not(isinstance(name, str)):
            name = str(name)
            
        self.n += 1
        self.all_tocs.append(tbeg)
        self.all_tocs_names.append(name)
        
        if name not in self.name_set:
            self.name_set.add(name)
            self.n_names +=1 
            self.max_name_len = max(self.max_name_len, len(name))

        if self.include_locs:
            caller = inspect.getframeinfo(inspect.stack()[1][0])
            
            toc_loc = ''
            if self.include_filename:
                filename = os.path.relpath(os.path.abspath(caller.filename), self.path_prefix)
                
                toc_loc += f'file {filename} '
            
            if self.include_lineno:
                toc_loc += f'line {caller.lineno} '

            if self.include_funname and caller.function != '<module>':
                toc_loc += f'in {caller.function} '
            
            toc_loc = toc_loc[:-1]
            
            self.all_tocs_locs.append(toc_loc)
            
        tend = time.perf_counter()
        dt = tend-tbeg
        self.all_tocs_record_time.append(dt)
        self.all_tocs_record_time_sum += dt
        
    def _get_recorded_time(self, idx):
        return self.all_tocs[idx+1]-(self.all_tocs[idx] + self.all_tocs_record_time[idx])

    @property
    def _total_time(self):
        return self.all_tocs[self.n]-self.all_tocs[0]-self.all_tocs_record_time_sum + self.all_tocs_record_time[self.n]
        
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
                
                out += f'{self._get_recorded_time(i):.8f} s'
                if self.include_locs:
                    out += f' at {self.all_tocs_locs[i]}'
                    
                out += '\n'
        
        else:
            
            d, first = self.to_dict(return_first_instance=True)
            for name, arr in d.items():
                
                if self.align_toc_names:
                    filler = ' ' * (self.max_name_len - len(name))
                else:
                    filler = ''
                
                if name != '':
                    out += f'{name}{filler}: '
                    
                out += f'{self.names_reduction(arr):.8f} s'
                if self.include_locs:
                    out += f' at {self.all_tocs_locs[first[name]]}'
                    
                out += '\n'
        out += '\n'
        out += f'Total: {self._total_time:.8f} s\n'

            
        return out
    
    def to_dict(
        self                                    ,
        return_first_instance : bool = False    ,
        names_reduction = None                  ,
    ):
        """
        Returns time measurements within a TimeTrain as a Python dictionnary

        Parameters
        ----------
        return_first_instance : bool, optional
            Whether to also return a dictionnary containing the index of the first occurence of every name, by default False

        """        
        
        dict_list = {}
        dict_first = {}
        for i, name in enumerate(self.all_tocs_names):
            
            t = self._get_recorded_time(i)
            
            toclist = dict_list.get(name)
            if toclist is None:
                dict_list[name] = [t]
                dict_first[name] = i 
            else:
                toclist.append(t)
                
        if names_reduction is None:
            names_reduction = self.names_reduction
        
        if names_reduction is None:
            res = {name: np.array(l) for name, l in dict_list.items()}
        else:
            res = {name: names_reduction(np.array(l)) for name, l in dict_list.items()}
            
        if return_first_instance:
            return res, dict_first
        else:
            return res
    
    def __len__(self):
        
        return self.n_names
        