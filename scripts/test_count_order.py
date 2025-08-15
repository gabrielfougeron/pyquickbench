import os
import numpy as np
import itertools
import pyquickbench

TT = pyquickbench.TimeTrain()

def brute_force(v,w):
    
    n = v.shape[0]
    m = w.shape[0]
    
    res = 0
    
    for i in range(n):
        for j in range(m):
            if v[i] <= w[j]:
                res += 1
                
    return res


def more_subtle(v,w):
    
    n = v.shape[0]
    m = w.shape[0]
    
    u = np.concatenate((v,w))
    idx_sort = np.argsort(u)
    
    mm = m
    
    res = 0
    
    for i in range(n+m-1):
        if idx_sort[i] < n:
            res += mm
        else:
            mm -= 1
            
    return res


n = 5
m = 5
p = 5
q = 5

v = np.random.random(n)
w = np.random.random(m)
y = np.random.random(p)
z = np.random.random(q)

assert brute_force(v,w) == more_subtle(v,w)

@TT.tictoc
def brute_force_list(*l):
    
    nvec = len(l)
    if nvec < 2:
        return
    
    shapes = [l[i].shape[0] for i in range(nvec)]
    res = 0
    
    ranges = [range(shapes[i]) for i in range(nvec)]
    
    for I in itertools.product(*ranges):
        
        test = True
        for i in range(nvec-1):
            test = test and l[i][I[i]] <= l[i+1][I[i+1]]

        if test:
            res += 1
            
    return res

assert brute_force(v,w) == brute_force_list(v,w)


def mores_subtle_list(*l):
    
    nvec = len(l)
    if nvec < 2:
        return
    
    shapes = [l[i].shape[0] for i in range(nvec)]
    
    nelem_tot = sum(shapes)
    
    cum_shapes = np.zeros(nvec, dtype=np.intp)
    cum_shapes[0] = shapes[0]
    for i in range(nvec-1):
        cum_shapes[i+1] = cum_shapes[i] + shapes[i+1]
    
    u = np.concatenate(l)
    idx_sort = np.argsort(u)
    print(idx_sort)
    
    idx_sorted_to_ivec = np.searchsorted(cum_shapes, idx_sort, side='right')
    
    idx_sort_vec = [np.empty(shapes[ivec]+1,dtype=np.intp) for ivec in range(nvec)]
    cur_idx = np.zeros(nvec,dtype=np.intp)
    for i in range(nelem_tot):
        ivec = idx_sorted_to_ivec[i]
        idx_sort_vec[ivec][cur_idx[ivec]] = i
        cur_idx[ivec] += 1
    for jvec in range(nvec):
        idx_sort_vec[jvec][shapes[ivec]] = nelem_tot
    idx_sort_vec.append(np.array([nelem_tot]))
    
    print(idx_sort_vec)
    
    cur_idx_min = np.zeros(nvec+1,dtype=np.intp)
    for jvec in range(nvec):
        while idx_sort_vec[jvec+1][cur_idx_min[jvec+1]] < idx_sort_vec[jvec][cur_idx_min[jvec]]:
            cur_idx_min[jvec+1] += 1

    cur_idx_max = np.zeros(nvec+1,dtype=np.intp)
    
    for jvec in range(nvec):
        while idx_sort_vec[jvec][cur_idx_max[jvec]] < idx_sort_vec[jvec+1][cur_idx_min[jvec+1]]:
            cur_idx_max[jvec] += 1
    

    # res = 0
    # 
    # for i in range(nelem_tot):
    #     
    #     ivec = idx_sorted_to_ivec[i]
    #     
    #     test = True
    #     for jvec in range(nvec):
    #         test = test and (idx_sort_vec[jvec][cur_idx_min[jvec]] < idx_sort_vec[jvec+1][cur_idx_min[jvec+1]])
    #          
    #     if test:
    #         res += np.prod(el_rem[1:])
    #     else:
    #         el_rem[ivec] -= 1
    #         
    #     cur_idx[ivec] += 1
    #       
    # return res
    #     
    
    
    
    

print(v)
print(w)
print(z)

ref = brute_force(v,w)
# assert ref == more_subtle(v,w)
# assert ref == brute_force_list(v,w)
# assert ref == mores_subtle_list(v,w)

# 
# print(brute_force_list(v,w))
# print(mores_subtle_list(v,w))

# ref = brute_force_list(v,w,z)
# assert ref == mores_subtle_list(v,w,z)


@TT.tictoc
def mores_subtle_list_new(*l):
    
    nvec = len(l)
    if nvec < 2:
        return
    
    shapes = [l[i].shape[0] for i in range(nvec)]
    
    nelem_tot = sum(shapes)
    
    cum_shapes = np.zeros(nvec, dtype=np.intp)
    cum_shapes[0] = shapes[0]
    for i in range(nvec-1):
        cum_shapes[i+1] = cum_shapes[i] + shapes[i+1]
    
    u = np.concatenate(l)
    idx_sort = np.argsort(u)
    print(idx_sort)
    
    idx_sorted_to_ivec = np.searchsorted(cum_shapes, idx_sort, side='right')
    print(idx_sorted_to_ivec)
    
    idx_sort_vec = [np.empty(shapes[ivec],dtype=np.intp) for ivec in range(nvec)]
    cur_idx = np.zeros(nvec,dtype=np.intp)
    for i in range(nelem_tot):
        ivec = idx_sorted_to_ivec[i]
        idx_sort_vec[ivec][cur_idx[ivec]] = i
        cur_idx[ivec] += 1
    
    for arr in idx_sort_vec:
        print(arr)
    
    # print(idx_sort_vec)
    
#     cur_idx_min = np.zeros(nvec+1,dtype=np.intp)
#     for jvec in range(nvec):
#         while idx_sort_vec[jvec+1][cur_idx_min[jvec+1]] < idx_sort_vec[jvec][cur_idx_min[jvec]]:
#             cur_idx_min[jvec+1] += 1
# 
#     cur_idx_max = np.zeros(nvec+1,dtype=np.intp)
#     
#     for jvec in range(nvec):
#         while idx_sort_vec[jvec][cur_idx_max[jvec]] < idx_sort_vec[jvec+1][cur_idx_min[jvec+1]]:
#             cur_idx_max[jvec] += 1
    # 
    res = 0
    for idx in itertools.combinations(range(nelem_tot), nvec):
        test = True
        for i in range(nvec):
            test = test and idx_sorted_to_ivec[idx[i]] == i
            
        if test:
            res += 1
            
    return res




    # res = 0
    # 
    # for i in range(nelem_tot):
    #     
    #     ivec = idx_sorted_to_ivec[i]
    #     
    #     test = True
    #     for jvec in range(nvec):
    #         test = test and (idx_sort_vec[jvec][cur_idx_min[jvec]] < idx_sort_vec[jvec+1][cur_idx_min[jvec+1]])
    #          
    #     if test:
    #         res += np.prod(el_rem[1:])
    #     else:
    #         el_rem[ivec] -= 1
    #         
    #     cur_idx[ivec] += 1
    #       
    # return res
    #     
    
    

    
n = 5
m = 5
p = 5
q = 5

v = np.random.random(n)
w = np.random.random(m)
y = np.random.random(p)
z = np.random.random(q)

print(f'{brute_force_list(v,w,y,z) = }')
    
print(mores_subtle_list_new(v,w,y,z))

print(TT)