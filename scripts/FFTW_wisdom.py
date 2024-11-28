import pyfftw
import numpy as np
import multiprocessing
import itertools
import os
import tqdm
import json

try:
    __PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))

    if ':' in __PROJECT_ROOT__:
        __PROJECT_ROOT__ = os.getcwd()

except (NameError, ValueError): 

    __PROJECT_ROOT__ = os.path.abspath(os.path.join(os.getcwd(),os.pardir))

    
timings_folder = os.path.join(__PROJECT_ROOT__,'examples','generated_files')

if not(os.path.isdir(timings_folder)):
    os.makedirs(timings_folder)
    
DP_Wisdom_file = os.path.join(timings_folder,"PYFFTW_wisdom.txt")

Load_wisdom = True
# Load_wisdom = False
# 
Write_wisdom = True
# Write_wisdom = False

if Load_wisdom:
    
    if os.path.isfile(DP_Wisdom_file):
        with open(DP_Wisdom_file,'r') as jsonFile:
            Wis_dict = json.load(jsonFile)
    
    wis = (
        Wis_dict["double"].encode('utf-8'),
        Wis_dict["single"].encode('utf-8'),
        Wis_dict["long"]  .encode('utf-8'),
    )

    pyfftw.import_wisdom(wis)

pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.set_keepalive_time(300000)

def setup_all(fft_type, nthreads, n, stride):
    
    simd_n = pyfftw.simd_alignment
    # 
    # planner_effort = ('FFTW_WISDOM_ONLY',)
    # planner_effort = ('FFTW_ESTIMATE',)
    # planner_effort = ('FFTW_MEASURE'   ,)
    # planner_effort = ('FFTW_PATIENT',)
    planner_effort = ('FFTW_EXHAUSTIVE',)
    
    # planner_effort = ('FFTW_EXHAUSTIVE', 'FFTW_WISDOM_ONLY')

    simd_n = pyfftw.simd_alignment
    
    if fft_type in ['fft', 'ifft']:
        m = n
        xdtype = 'complex128'
        ydtype = 'complex128'
    elif fft_type in ['rfft','irfft']:
        m = n//2 + 1
        xdtype = 'float64'
        ydtype = 'complex128'
        
    if fft_type in ['fft', 'rfft']:
        direction = 'FFTW_FORWARD'
    elif fft_type in ['ifft','irfft']:
        direction = 'FFTW_BACKWARD'
        m, n = n, m
        xdtype, ydtype = ydtype, xdtype
        
    x = pyfftw.empty_aligned((n, stride), dtype=xdtype, n=simd_n)
    y = pyfftw.empty_aligned((m, stride), dtype=ydtype, n=simd_n)
    
    fft_object = pyfftw.FFTW(x, y, axes=(0, ), direction=direction, flags=planner_effort, threads=nthreads, planning_timelimit=None)

    return fft_object
    
    
all_fft_types = [
    'fft',
    'ifft',
    'rfft',
    'irfft',
]

all_nthreads = [
    1, 
    # multiprocessing.cpu_count()//2
]

base_fac_list = [128]
# base_fac_list = [2*n+1 for n in range(13)]
# base_fac_list = list(range(100))


n_exp_min = 0
n_exp_max = 10
# n_exp_max = 6

all_strides = [1]
# all_strides = [1, 6]

all_sizes = np.array([base_fac * 2**n_exp for n_exp in range(n_exp_min,n_exp_max+1) for base_fac in base_fac_list])
    
total_it = len(all_fft_types) * len(all_nthreads) * len(all_sizes) * len(all_strides)
    
all_custom = []    
with tqdm.tqdm(total=total_it) as progress_bar:
    for fft_type, nthreads, n, stride in itertools.product(all_fft_types, all_nthreads, all_sizes, all_strides):
        
        # print(f'{fft_type = } {nthreads = } {n = }')
        
        all_custom.append(setup_all(fft_type, nthreads, n, stride))
        
        if Write_wisdom:    
            wis = pyfftw.export_wisdom()
            
            Wis_dict = {
                "double": wis[0].decode('utf-8') , 
                "single": wis[1].decode('utf-8') ,
                "long":   wis[2].decode('utf-8') ,
            }
                
            with open(DP_Wisdom_file, "w") as jsonFile:
                jsonString = json.dumps(Wis_dict, indent=4, sort_keys=False)
                jsonFile.write(jsonString)
            
        progress_bar.update(1)
    
print("Done!")