import sys
import os
import math
import numpy as np

import pyquickbench
import GUI


try:
    __PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))

    if ':' in __PROJECT_ROOT__:
        __PROJECT_ROOT__ = os.getcwd()

except (NameError, ValueError): 

    __PROJECT_ROOT__ = os.path.abspath(os.path.join(os.getcwd(),os.pardir,os.pardir))

sys.path.append(__PROJECT_ROOT__)

# bench_root = os.path.join(__PROJECT_ROOT__, "AI_bench", "AR")
bench_root = os.path.join(__PROJECT_ROOT__, "AI_bench", "AR_2")

# compare_intent = {}
compare_intent = {"t5xxl_prompt" : "group"}
# compare_intent = {"lora_name" : "group"}

rank_assign = pyquickbench.ManualRankAssign(bench_root, k=2, compare_intent=compare_intent)

img_compare_GUI = GUI.ImageCompareGUI(rank_assign)
img_compare_GUI()

print()

# compare_intent = {"lora_strength" : "group"}
compare_intent = {"lora_name" : "group"}


order, v = rank_assign.get_order(compare_intent = compare_intent)

for rank, d in enumerate(order):
    
    print(f'{rank = }')
    print(f'P-L weight = {v[rank]}')
    
    for key, val in d.items():
        
        print(key, ':', val)
        
    print() 