import os
import numpy as np
import time
import inspect
import warnings
import functools

import typing
import numpy.typing

from pyquickbench._defaults import *

class TimeTrain():
    """ Records elapsed time between interest points in code
    
    See :ref:`sphx_glr__build_auto_examples_tutorial_10-TimeTrains.py` for usage example.\n
    
    """
    
    def __init__(
        self                                                    ,
        path_prefix         : typing.Union[str , None] = None   ,
        include_locs        : typing.Union[bool, None] = None   ,
        include_filename    : bool = True                       ,
        include_lineno      : bool = True                       ,
        include_funname     : bool = True                       ,
        name                : str  = ''                         ,
        align_toc_names     : bool = True                       ,
        names_reduction     : typing.Union[str , None] = None   ,
        global_tictoc_sync  : bool = True                       ,
        ignore_names        : typing.Iterable[str]  = None      ,
        relative_timings    : bool = False                      ,
    ):
        """ Returns a TimeTrain

        Parameters
        ----------
        path_prefix : :class:`python:str` | :data:`python:None`, optional
            Path relative to which other paths are to be understood, by default :data:`python:None`.
        include_locs : :class:`python:bool` | None, optional
            Whether to include locations in code when printing the TimeTrain, by default :data:`python:None`.
        include_filename : :class:`python:bool`, optional
            Whether to include the file name in locations in code when printing the TimeTrain, by default :data:`python:True`.
        include_lineno : :class:`python:bool`, optional
            Whether to include the line number in locations in code when printing the TimeTrain, by default :data:`python:True`.
        include_funname : :class:`python:bool`, optional
            Whether to include the function name in locations in code when printing the TimeTrain, by default :data:`python:True`.
        name : :class:`python:str`, optional
            Name of the TimeTrain, by default ``''``.
        align_toc_names : :class:`python:bool`, optional
            Whether to align toc names when printing the TimeTrain, by default :data:`python:True`.
        names_reduction : :class:`python:str` | :data:`python:None`, optional
            Reduction to be applied to tocs that share the same name, by default :data:`python:None`.
        global_tictoc_sync : :class:`python:bool`, optional
            Set to :data:`python:True` to use a common shared name for synchronization in :meth:`pyquickbench.TimeTrain.tictoc` or to :data:`python:False` to use a specific name for every decorated function. By default :data:`python:True`.
        ignore_names : :class:`python:str` | :data:`python:None`, optional
            Names to be ignored by :func:`python:print` and :meth:`pyquickbench.TimeTrain.to_dict`.
            By default :data:`python:None`.
        relative_timings : :class:`python:bool`, optional
            Whether to display relative timings when printing the TimeTrain.
            Relative timings only sum up to 100% if ``names_reduction`` is :data:`python:None` or ``\"sum\"``.
            By default :data:`python:False`.

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
        self.context_depth = 1
        self.global_tictoc_sync = global_tictoc_sync
        self.sum_notignored_time = 0
        
        if ignore_names is None:
            self.ignore_names = default_TimeTrain_ignore_names
        else:
             self.ignore_names = ignore_names
        
        self.relative_timings = relative_timings
        
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
            self.names_reduction_key = None
            self.names_reduction = None   
        else:
            self.names_reduction_key = names_reduction
            self.names_reduction = all_reductions.get(names_reduction)
            if self.names_reduction is None:
                raise ValueError(f'Unknown reduction {names_reduction}')
        
        if self.include_locs and (self.names_reduction is not None):
            warnings.warn("include_locs and names_reduction were both enabled. Only the first location will be displayed for every name.")
        
        if self.relative_timings:
            
            if self.names_reduction_key not in [None, "sum"]:
                
                warnings.warn(f"Reduction {self.names_reduction_key} combined with \"relative_timings = True\" will lead to reported timings not adding up to 100%.")
        
        # This line goes at the very end for more precise measurements
        self.all_tocs = [time.perf_counter()]
        
    def toc(
        self                ,
        name    : str = ''  ,
    ):
        """
        Records a new wagon in the TimeTrain.
        
        See :ref:`sphx_glr__build_auto_examples_tutorial_10-TimeTrains.py` for usage example.  

        Parameters
        ----------
        name : :class:`python:str`, optional
            Name of the wagon. This name is used as a key in :meth:`pyquickbench.TimeTrain.to_dict` and in calls to :func:`python:print`.\n
            By default ``''``.
        """        
        
        tbeg = time.perf_counter()

        if name is None:
            name = str(self.n)
        elif not(isinstance(name, str)):
            name = str(name)
            
        ignore = name in self.ignore_names
            
        self.n += 1
        self.all_tocs.append(tbeg)
        self.all_tocs_names.append(name)

        if name not in self.name_set:
            self.name_set.add(name)
            self.n_names +=1 
            if not ignore:
                self.max_name_len = max(self.max_name_len, len(name)+1)

        if self.include_locs:
            caller = inspect.getframeinfo(inspect.stack(context=self.context_depth)[self.context_depth][0])
            
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
    def total_time(self):
        """
        Returns the total time recorded by the TimeTrain so far.
        
        Note that the time reported by :meth:`pyquickbench.TimeTrain.total_time` might differ significantly from wall clock time because pyquickbench tries to exclude its inner workings from the timing. 
         
        """
        
        res = 0.
    
        for i in range(self.n):

            if self.all_tocs_names[i] not in self.ignore_names:        
                res += self._get_recorded_time(i)
        return res
    
    def __str__(self): 
        
        total_time = self.total_time
        out = ''
        
        if self.name == '':
            out += 'TimeTrain results:' + os.linesep
        else:
            out += f'TimeTrain {self.name} results:' + os.linesep
            
        if self.names_reduction is None:
            
            out += os.linesep
            
            for i in range(self.n):

                name = self.all_tocs_names[i]    
                
                if name not in self.ignore_names:        
                    
                    if self.align_toc_names:
                        filler = ' ' * (self.max_name_len - len(name))
                    else:
                        filler = ''
                    
                    if name != '':
                        out += f'{name}{filler}: '
                    
                    time = self._get_recorded_time(i)
                    
                    if self.relative_timings:
                        out += f'{100*time/total_time:.8f} %'
                    else:
                        out += f'{time:.8f} s'
                        
                    if self.include_locs:
                        out += f' at {self.all_tocs_locs[i]}'
                        
                    out += os.linesep
        
        else:
            
            out += f'Reduction: {self.names_reduction_key}' + os.linesep + os.linesep
            
            d, first = self.to_dict(return_first_instance=True)
            for name, arr in d.items():
                
                if self.align_toc_names:
                    filler = ' ' * (self.max_name_len - len(name))
                else:
                    filler = ''
                
                if name != '':
                    out += f'{name}{filler}: '
                
                time = self.names_reduction(arr)
                    
                if self.relative_timings:
                    out += f'{100*time/total_time:.8f} %'
                else:
                    out += f'{time:.8f} s'
                    
                if self.include_locs:
                    out += f' at {self.all_tocs_locs[first[name]]}'
                    
                out += os.linesep
                
        if (self.n) > 0:                
            out += os.linesep
            
        out += f'Total: {total_time:.8f} s'+os.linesep

        return out
    
    def __repr__(self):
        return self.__str__()

    def __format__(self, format_spec):
        return self.__str__()
        
    def to_dict(
        self                                                                            ,
        return_first_instance   : bool                                      = False     ,
        names_reduction         : typing.Union[typing.Callable, str, None]  = "default" ,
    ):
        """
        Returns time measurements within a TimeTrain as a Python dictionary

        See :ref:`sphx_glr__build_auto_examples_tutorial_10-TimeTrains.py` for usage example.    

        Parameters
        ----------
        return_first_instance : :class:`python:bool`, optional
            Whether to also return a dictionary containing the index of the first occurrence of every name, by default :data:`python:False`.
        names_reduction : :term:`python:callable` | :class:`python:str` | :data:`python:None`, optional
            Optionally overrides the TimeTrain's reduction.
            Set to ``"default"`` to not override reduction.
            By default ``"default"``.
            
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
                
        if names_reduction == "default":
            names_reduction = self.names_reduction

        if names_reduction is None:
            res = {name: np.array(l) for name, l in dict_list.items() if name not in self.ignore_names}
        else:
            res = {name: names_reduction(np.array(l)) for name, l in dict_list.items() if name not in self.ignore_names}
            
        if return_first_instance:
            return res, dict_first
        else:
            return res
    
    def __len__(self):
        
        return self.n_names
        
    @property
    def tictoc(self):
        """
        Decorates a function to record a new wagon in the TimeTrain before and after each call.\n
        By default, the first wagon (used for synchronization) is ignored.
        
        See :ref:`sphx_glr__build_auto_examples_tutorial_10-TimeTrains.py` for usage example. 
        
        Parameters
        ----------
        name :  :class:`python:str` | :data:`python:None`, optional
            Optionally overrides the wrapped function's name.\n
            By default :data:`python:None`.
            
        """        

        def decorator(fun, name=None):
            
            @functools.wraps(fun)
            def wrapper(*args, **kwargs):
                
                # Subtle python compiler behavior
                if name is None:
                    the_name = wrapper.__name__
                else:
                    the_name = name
                
                context_depth_prev = self.context_depth
                self.context_depth = 2
                
                if self.global_tictoc_sync:
                    self.toc(tictoc_sync_name)
                else:
                    self.toc(f'{tictoc_sync_name}_{the_name}')    
                
                fun(*args, **kwargs)
                
                self.toc(the_name)
                self.context_depth = context_depth_prev
            
            return wrapper
                
        return decorator   
            