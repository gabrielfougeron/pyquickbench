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
bench_root = os.path.join(__PROJECT_ROOT__, "AI_bench", "sarah_x4")

# compare_intent = {}
compare_intent = {"t5xxl_prompt" : "group"}
# compare_intent = {"lora_name" : "group"}

restrict_values = {
    "lora_name" : [
        # 'sarah\\saraheveliina_alpha16_dev_rank2_bf16-step04000.safetensors' ,
        'sarah\\saraheveliina_double_rank4_bf16-step04000.safetensors'      ,
        # 'sarah\\saraheveliina_single_noT5_rank4_bf16-step04000.safetensors' ,
        # 'sarah\\saraheveliina_single_rank4_bf16-step04000.safetensors'       ,
    ]
}

rank_assign = pyquickbench.ManualRankAssign(
    bench_root, k=2,
    compare_intent=compare_intent,
    restrict_values=restrict_values
)

img_compare_GUI = GUI.ImageCompareGUI(rank_assign)
img_compare_GUI()

print()

# compare_intent = {"lora_strength" : "group"}

compare_intent = {
    # "lora_name" : "group"       ,
    # "lora_strength" : "group"   ,
    "sampler_steps" : "group"   ,
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