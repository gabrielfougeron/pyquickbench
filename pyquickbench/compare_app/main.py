import sys
import os
import math
import numpy as np

import pyquickbench
import GUI


# try:
#     __PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
# 
#     if ':' in __PROJECT_ROOT__:
#         __PROJECT_ROOT__ = os.getcwd()
# 
# except (NameError, ValueError): 
# 
#     __PROJECT_ROOT__ = os.path.abspath(os.path.join(os.getcwd(),os.pardir,os.pardir))
# 
# sys.path.append(__PROJECT_ROOT__)

# bench_root = '/mnt/e/ComfyUI/ComfyUI/output/cli/Marin_dev_all'
# bench_root = '/mnt/e/ComfyUI/ComfyUI/output/cli/sarah_x4'
# bench_root = '/mnt/e/ComfyUI/ComfyUI/output/cli/ambiebambii_dev_all'
# bench_root = '/mnt/e/ComfyUI/ComfyUI/output/cli/ambiebambii_nocropface_T5_dev_all'
# bench_root = '/mnt/e/ComfyUI/ComfyUI/output/cli/ambiebambii_nocropface_T5_dev_all'
# bench_root = '/mnt/e/ComfyUI/ComfyUI/output/cli/ambiebambii_T5_vs_noT5'
bench_root = '/mnt/e/ComfyUI/ComfyUI/output/cli/ambiebambii_nocropface_noT5_dev_all'
# bench_root = '/mnt/e/ComfyUI/ComfyUI/output/cli/ambiebambii_krea'

compare_intent = {}
# compare_intent = {"t5xxl_prompt" : "group"}
# compare_intent = {"lora_name" : "group"}

# restrict_values = {}
restrict_values = {
    # "t5xxl_prompt" : [
    #     "A blonde young woman on her knees on her bed." ,
    #     "A blonde young woman wearing taking a selfie." ,
    #     "A blonde young woman playing beach volley."    ,
    # ],
    # "lora_name" : [
    #     r"training\ambiebambii_nocropface_T5_dev_all\lora_rank32_bf16-step04000.safetensors",
    #     r"training\ambiebambii_nocropface_T5_dev_all\lora_rank32_bf16-step05000.safetensors",
    #     r"training\ambiebambii_nocropface_T5_dev_all\lora_rank32_bf16-step06000.safetensors",
    #     r"training\ambiebambii_nocropface_T5_dev_all\lora_rank32_bf16-step06523.safetensors",
    #     r"training\ambiebambii_nocropface_T5_dev_all\lora_rank32_bf16-step07046.safetensors",
    #     r"training\ambiebambii_nocropface_T5_dev_all\lora_rank32_bf16-step07569.safetensors",
    #     r"training\ambiebambii_nocropface_T5_dev_all\lora_rank32_bf16-step08092.safetensors",
    #     r"training\ambiebambii_nocropface_T5_dev_all\lora_rank32_bf16-step08615.safetensors",
    # ],
}

bench_filename = os.path.join(bench_root, "bench.npz")

if not os.path.isfile(bench_filename):
    raise ValueError(f"Benchmark file {bench_filename} not found.")

benchfile_shape, all_vals = pyquickbench.run_benchmark(
    filename = bench_filename       ,
    return_array_descriptor = True  ,
    StopOnExcept = True             ,
)

rank_assign = pyquickbench.ManualRankAssign(
    benchfile_shape = benchfile_shape,
    all_vals = all_vals,
    bench_root = bench_root,
    k=2,
    compare_intent = compare_intent,
    restrict_values = restrict_values,
)


rank_assign.print_restrict_bench()


img_compare_GUI = GUI.ImageCompareGUI(rank_assign)
img_compare_GUI()



print()

# compare_intent = {"lora_strength" : "group"}

compare_intent = {
    "lora_name" : "group"       ,
    "lora_strength" : "group"   ,
    # "unet_name" : "group"   ,
    "sampler_steps" : "group"   ,
    "flux_guidance" : "group"   ,
    "function" : "group"   ,
    # "t5xxl_prompt" : "group"   ,
}



# compare_intent = None


n_votes, order, v = rank_assign.get_order(compare_intent = compare_intent)

print("Total number of votes :", rank_assign.best_count.sum())
print("Total number of votes in comparison :", n_votes)
print()

for i, d in enumerate(order):
    
    rank = v.shape[0]-i
    
    print(f'{rank = }')
    print(f'P-L weight = {v[i]}')
    
    for key, val in d.items():
        
        print(key, ':', val)
        
    print() 