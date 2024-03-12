import matplotlib as mpl
import numpy as np

default_ax_name = "n"
fun_ax_name = "function"
repeat_ax_name = "repeat"
out_ax_name = "out"

default_color_list = list(mpl.colors.TABLEAU_COLORS)
default_color_list.append(mpl.colors.BASE_COLORS['b'])
default_color_list.append(mpl.colors.BASE_COLORS['g'])
default_color_list.append(mpl.colors.BASE_COLORS['r'])
default_color_list.append(mpl.colors.BASE_COLORS['m'])
default_color_list.append(mpl.colors.BASE_COLORS['k'])

default_linestyle_list = [
     'solid'                        ,   # Same as (0, ()) or '-'
     'dotted'                       ,   # Same as (0, (1, 1)) or ':'
     'dashed'                       ,   # Same as '--'
     'dashdot'                      ,   # Same as '-.'
     (0, (1, 10))                   ,   # loosely dotted   
     (0, (1, 1))                    ,   # dotted
     (0, (1, 1))                    ,   # densely dotted
     (5, (10, 3))                   ,   # long dash with offset
     (0, (5, 10))                   ,   # loosely dashed
     (0, (5, 5))                    ,   # dashed
     (0, (5, 1))                    ,   # densely dashed
     (0, (3, 10, 1, 10))            ,   # loosely dashdotted
     (0, (3, 5, 1, 5))              ,   # dashdotted
     (0, (3, 1, 1, 1))              ,   # densely dashdotted
     (0, (3, 5, 1, 5, 1, 5))        ,   # dashdotdotted
     (0, (3, 10, 1, 10, 1, 10))     ,   # loosely dashdotdotted
     (0, (3, 1, 1, 1, 1, 1))        ,   # densely dashdotdotted                    
]

default_pointstyle_list = [
    None,   
    "."	, # m00 point
    "o"	, # m02 circle
    "v"	, # m03 triangle_down
    "^"	, # m04 triangle_up
    "<"	, # m05 triangle_left
    ">"	, # m06 triangle_right
    "1"	, # m07 tri_down
    "2"	, # m08 tri_up
    "3"	, # m09 tri_left
    "4"	, # m10 tri_right
    "8"	, # m11 octagon
    "s"	, # m12 square
    "p"	, # m13 pentagon
    "P"	, # m23 plus (filled)
    "*"	, # m14 star
    "h"	, # m15 hexagon1
    "H"	, # m16 hexagon2
    "+"	, # m17 plus
    "x"	, # m18 x
    "X"	, # m24 x (filled)
    "D"	, # m19 diamond
    "d"	, # m20 thin_diamond
]

def default_setup(n):
    return {default_ax_name: n}

Legend_markersize = 10

Legend_bland_color = 'k'
Legend_bland_linestyle = 'solid'
Legend_bland_pointstyle = None

def ma_logavg(obj, axis=None, keepdims=np._NoValue):

    log = np.ma.log(obj)
    avg = np.ma.mean(log, axis=axis, keepdims=keepdims)
    
    return np.exp(avg)

def ma_idx_el(obj, idx , axis=None, keepdims=np._NoValue):
    
    if (keepdims is np._NoValue) or (not keepdims):        
        return np.take(obj, idx, axis=axis)
    else:
        return np.take(obj, [idx], axis=axis)

def ma_first_el(obj, axis=None, keepdims=np._NoValue):
    
    return ma_idx_el(obj, 0, axis=axis, keepdims=keepdims)

def ma_last_el(obj, axis=None, keepdims=np._NoValue):
    
    if axis is None:
        i = np.prod(obj.shape)-1
    else:
        i = obj.shape[axis]-1
        
    return ma_idx_el(obj, i, axis=axis, keepdims=keepdims)

def ma_random_el(obj, axis=None, keepdims=np._NoValue):
    
    if axis is None:
        n = np.prod(obj.shape)
    else:
        n = obj.shape[axis]
    
    i = np.random.randint(n)
    return ma_idx_el(obj, i, axis=axis, keepdims=keepdims)

all_reductions = {
    "avg"       : np.ma.mean    ,
    "min"       : np.ma.min     , 
    "max"       : np.ma.max     ,
    "median"    : np.ma.median  ,
    "sum"       : np.ma.sum     ,
    "logavg"    : ma_logavg     ,
    "first_el"  : ma_first_el   ,
    "last_el"   : ma_last_el    ,
    "random_el" : ma_random_el  ,
}

all_plot_intents = [
    'single_value'      ,
    'points'            ,
    'same'              ,
    'curve_color'       ,
    'curve_linestyle'   ,
    'curve_pointstyle'  ,
    'subplot_grid_x'    ,
    'subplot_grid_y'    ,
]

all_plot_intents.extend([f'reduction_{name}' for name in all_reductions])

