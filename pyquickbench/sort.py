""" NOT FOR PERFORMANCE!

"""

import numpy as np

__all__ = [
    'merge_sort'            ,
    'insertion_sort'        ,
    'heap_sort'             ,
    'merge_insertion_sort'  ,
    'quick_sort'            ,
    'binary_insertion_sort' ,
]

default_cmp_lt = lambda a,b:a<b

def merge_sort(arr, cmp_lt=default_cmp_lt):
    
    merge_sort_ann(arr, 0, len(arr)-1, cmp_lt)

def merge_sort_ann(arr, l, r, cmp_lt=default_cmp_lt):
    
    if (l < r):

        m = l + (r - l) // 2

        merge_sort_ann(arr, l, m, cmp_lt)
        merge_sort_ann(arr, m + 1, r, cmp_lt)

        merge(arr, l, m, r, cmp_lt)
        
def merge(arr, start, mid, end, cmp_lt=default_cmp_lt):
    
    start2 = mid + 1

    if cmp_lt(arr[mid], arr[start2]):
        return

    while (start <= mid and start2 <= end):

        if cmp_lt(arr[start], arr[start2]):
            start += 1
            
        else:
            value = arr[start2]
            index = start2

            while (index != start):
                arr[index] = arr[index - 1]
                index -= 1

            arr[start] = value

            start += 1
            mid += 1
            start2 += 1

def insertion_sort(arr, cmp_lt=default_cmp_lt):
    
    n = len(arr)
    
    for i in range(1, n):
        key = arr[i]
        j = i-1
        while j >= 0 and cmp_lt(key, arr[j]):
            arr[j+1] = arr[j]
            j -= 1
        arr[j+1] = key
        
def binary_search(arr, val, start, end, cmp_lt=default_cmp_lt):
    
    if start == end:
        if cmp_lt(val, arr[start]):
            return start
        else:
            return start + 1

    elif start > end:
        return start

    else:
        mid = (start + end) // 2
        if cmp_lt(arr[mid], val):
            return binary_search(arr, val, mid + 1, end, cmp_lt)
        else:
            return binary_search(arr, val, start, mid - 1, cmp_lt)

def binary_insertion_sort(arr, cmp_lt=default_cmp_lt):
    
    for i in range(1, len(arr)):
        
        val = arr[i]
        j = binary_search(arr, val, 0, i - 1, cmp_lt)
        
        for k in range(i-1,j-1,-1):
            arr[k+1] = arr[k]
                 
        arr[j] = val


def heapify(arr, n, i, cmp_lt=default_cmp_lt):
    
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2

    if l < n and cmp_lt(arr[i], arr[l]):
        largest = l

    if r < n and cmp_lt(arr[largest], arr[r]):
        largest = r

    if largest != i:
        (arr[i], arr[largest]) = (arr[largest], arr[i])

        heapify(arr, n, largest, cmp_lt)

def heap_sort(arr, cmp_lt=default_cmp_lt):
    
    n = len(arr)

    for i in range(n // 2, -1, -1):
        heapify(arr, n, i, cmp_lt)

    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0, cmp_lt)

def binary_insert(arr, x, cmp_lt=default_cmp_lt):
    
    possible_positions = range(len(arr) + 1)
    L, R = 0, possible_positions[-1]
    while len(possible_positions) > 1:
        m = (L + R)//2
        if cmp_lt(x, arr[m]):
            R = m
        else:
            L = m+1
        possible_positions = [p for p in possible_positions if L <= p <= R]
    return possible_positions[0]

def merge_insertion_sort(arr, cmp_lt=default_cmp_lt):
    """
    cf https://en.wikipedia.org/wiki/Merge-insertion_sort
    
    """
    if len(arr) <= 1:
        return
    pairs = []
    for x1, x2 in zip(arr[::2], arr[1::2]):
        if cmp_lt(x1, x2):
            pairs.append((x1, x2))
        else:
            pairs.append((x2, x1))

    merge_insertion_sort(pairs, cmp_lt=lambda a,b : cmp_lt(a[1], b[1]))
    res = [x2 for x1, x2 in pairs]
    remaining = pairs[:]
    if len(arr) % 2 == 1:
        remaining.append((arr[-1], "END"))
    res.insert(0, remaining.pop(0)[0])
    
    buckets = [2, 2]
    power_of_2 = 4
    while sum(buckets) < len(remaining):
        power_of_2 *= 2
        buckets.append(power_of_2 - buckets[-1])
    reordered = []
    last_index = 0
    for bucket in buckets:
        reordered += reversed(remaining[last_index:last_index+bucket])
        last_index += bucket
    for y, x in reordered:
        if x == "END":
            res.insert(binary_insert(res, y, cmp_lt), y)
        else:
            res.insert(binary_insert(res[:res.index(x)], y, cmp_lt), y)
        
    # Stupid but all other sorting algorithms are in-place
    for i in range(len(arr)):
        arr[i] = res[i]   

def partition(arr, low, high, cmp_lt=default_cmp_lt):
    
    i = (low - 1)      
    pivot = arr[high]  

    for j in range(low, high):
        if cmp_lt(arr[j], pivot):

            i = i + 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

def quick_sort_ann(arr, low, high, cmp_lt=default_cmp_lt):
    
    if low < high:
        
        pi = partition(arr, low, high, cmp_lt)
        
        quick_sort_ann(arr, low, pi - 1, cmp_lt)
        quick_sort_ann(arr, pi + 1, high, cmp_lt)
        
def quick_sort(arr, cmp_lt=default_cmp_lt):
    quick_sort_ann(arr, 0, len(arr)-1, cmp_lt)
    
def argsort(arr, sort=merge_sort, cmp_lt=default_cmp_lt):
    arr_int = np.array(range(len(arr)), dtype=np.intp)
    arg_cmp_lt = lambda i,j:cmp_lt(arr[i],arr[j])
    sort(arr_int, arg_cmp_lt)
    return arr_int
    