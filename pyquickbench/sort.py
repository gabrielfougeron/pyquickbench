""" NOT FOR PERFORMANCE!

"""

import numpy as np

default_cmp_lt = lambda a,b:a<b

def merge_sort(arr, cmp_lt=default_cmp_lt):
    
    merge_sort_ann(arr, 0, len(arr)-1, cmp_lt)

def merge_sort_ann(arr, l, r, cmp_lt):
    
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

def argsort(arr, sort=merge_sort, cmp_lt=default_cmp_lt):
    arr_int = np.array(range(len(arr)), dtype=np.intp)
    arg_cmp_lt = lambda i,j:cmp_lt(arr[i],arr[j])
    sort(arr_int, arg_cmp_lt)
    return arr_int
    