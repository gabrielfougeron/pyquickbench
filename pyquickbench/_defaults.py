from pyquickbench._utils import (
    PhonyProcessPoolExecutor    ,
)

import concurrent.futures
import matplotlib as mpl

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
    ","	, # m01 pixel
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

all_plot_intents = [
    'single_value'      ,
    'points'            ,
    'same'              ,
    'curve_color'       ,
    'curve_linestyle'   ,
    'curve_pointstyle'  ,
    'subplot_grid_x'    ,
    'subplot_grid_y'    ,
    'reduction_avg'     ,
    'reduction_min'     ,
    'reduction_max'     ,
]

AllPoolExecutors = {
    "phony"         :   PhonyProcessPoolExecutor                ,
    "thread"        :   concurrent.futures.ThreadPoolExecutor   ,
    "process"       :   concurrent.futures.ProcessPoolExecutor  ,
}

def default_setup(n):
    return {'n': n}
