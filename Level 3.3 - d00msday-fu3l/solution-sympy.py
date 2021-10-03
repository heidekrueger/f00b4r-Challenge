## My second attempt using sympy
## Works, but not in the foobar challenge sandbox
## as it doesn't allow the sympy library


import math
import numpy as np
from fractions import gcd
from functools import reduce
import sympy as sp

def normalize_matrix(m):
    M = np.array(m, dtype=np.int32)
    rowsums = M.sum(axis=1)
    M_entries = M.flatten()[M.flatten()>0]
    denom = rowsums[rowsums>0].prod()

    for i in range(len(m)):        
        if rowsums[i] > 0:
             M[i,:] = M[i,:] * denom / rowsums[i] 
    M = sp.Matrix(M) / denom
    return M , denom

def is_terminal_state(row):
    return all([col == 0 for col in row])

def solution(m):
    n = len(m)
    terminal_states = [i for i, row in enumerate(m) if is_terminal_state(row)]
    if len(terminal_states) == n:
        return [1] + [0] * (n-1) + [1]

    M,d = normalize_matrix(m)
    inv =(sp.eye(n)-M).inv()[0,terminal_states]
    nums, denoms = zip(*[e.as_numer_denom() for e in inv])
    ### NOTE: the next line buggy, we need lcm rather than max
    ### this is fixed in the final Fractional solution, but I won't
    ### edit this file after the fact to include custom lcm implementations again.
    max_denom = max(denoms) 
    nums = (np.array(nums) * max_denom / np.array(denoms))
    
    return nums.tolist() + [max_denom]

    
    
input1 = [[0, 2, 1, 0, 0], [0, 0, 0, 3, 4], [0, 0, 0, 0, 0], [0, 0, 0, 0,0], [0, 0, 0, 0, 0]]
input2 = [[0, 1, 0, 0, 0, 1], [4, 0, 0, 3, 2, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]

print(solution(input1))