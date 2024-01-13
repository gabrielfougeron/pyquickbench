import concurrent.futures
import math
import numba as nb
import numpy as np

# import sys
# import os
# try:
#     __PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,os.pardir))
# 
#     if ':' in __PROJECT_ROOT__:
#         __PROJECT_ROOT__ = os.getcwd()
# 
# except (NameError, ValueError): 
# 
#     __PROJECT_ROOT__ = os.path.abspath(os.path.join(os.getcwd(),os.pardir,os.pardir))
# 
# sys.path.append(__PROJECT_ROOT__)
# import pyquickbenck


numba_opt_dict = {
    'nopython':True     ,
    'cache':True        ,
    'fastmath':True     ,
    'nogil':True        ,
}


PRIMES = [
    112272535095293,
    112582705942171,
    112272535095293,
    115280095190773,
    115797848077099,
    1099726899285419]

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return 0.

    maxval = math.ceil(math.sqrt(n)+1)
    # maxval = n
    for i in range(2, maxval):
        if n % i == 0:
            return 0.
    
    return 1.

is_prime_nb = nb.jit(is_prime,**numba_opt_dict)

def is_prime_nb_wrapped(n):
    m = n+3
    
    toto = is_prime_nb(n)
    
    return 1.-toto
    
def execution_wrapper(fun, arr, idx, n_repeat, *args, **kwargs):
    
    for i in range(n_repeat):
        # print(i, n_repeat)
        arr[idx, i] = fun(*args, **kwargs)

n_repeat = 100000   
all_out_vals = np.zeros((len(PRIMES), n_repeat))


def main(args):
    with concurrent.futures.ThreadPoolExecutor(6) as executor:
    # with concurrent.futures.ProcessPoolExecutor(6) as executor:
        
        for i, n in enumerate(args):
            
            # future = executor.submit(is_prime, n)
            # future = executor.submit(is_prime_nb, n)
            # future = executor.submit(is_prime_nb_wrapped, n)
            future = executor.submit(execution_wrapper, is_prime_nb, all_out_vals, i, n_repeat, n)


if __name__ == '__main__':
    main(PRIMES)