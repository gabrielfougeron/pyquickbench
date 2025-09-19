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
# bench_root = os.path.join(__PROJECT_ROOT__, "AI_bench", "sarah_x4")
# bench_root = os.path.join(__PROJECT_ROOT__, "AI_bench", "meadows")
bench_root = os.path.join(__PROJECT_ROOT__, "AI_bench", "Tristan_dev_all")

# compare_intent = {}
# compare_intent = {"t5xxl_prompt" : "group"}
# compare_intent = {"lora_name" : "group"}

# restrict_values = {
#     "unet_name" : [
#         'flux\\flux1-dev.safetensors',
#         # 'flux\\flux1-krea-dev.safetensors'
#     ]
# }

# restrict_values = None

rank_assign = pyquickbench.ManualRankAssign(
    bench_root, k=2,
    # compare_intent = compare_intent,
    # restrict_values = restrict_values,
)

# print(rank_assign.benchfile_shape)



img_compare_GUI = GUI.ImageCompareGUI(rank_assign)
img_compare_GUI()



print()

# compare_intent = {"lora_strength" : "group"}

compare_intent = {
    # "lora_name" : "group"       ,
    "lora_name" : "group"       ,
    "lora_strength" : "group"   ,
    "unet_name" : "group"   ,
    "sampler_steps" : "group"   ,
    "flux_guidance" : "group"   ,
    "function" : "group"   ,
    # "t5xxl_prompt" : "group"   ,
}



# compare_intent = None


n_votes, order, v = rank_assign.get_order(compare_intent = compare_intent)

print("Total number of votes :", n_votes)
print()

for i, d in enumerate(order):
    
    rank = v.shape[0]-i
    
    print(f'{rank = }')
    print(f'P-L weight = {v[i]}')
    
    for key, val in d.items():
        
        print(key, ':', val)
        
    print() 