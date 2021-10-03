## My first attempt at solving the challenge 
## this is quick and compact (especially in py3)
## but fails to solve the challenge as
## limited float/double precision isn't enough
## For at least one test case, the hack at the end
## of getting rationals from floats is insufficient.

import math
import numpy as np
from fractions import gcd
from functools import reduce

def normalize_matrix(m):
    M = np.array(m, dtype=np.double)
    rowsums = M.sum(axis=1)
    M_entries = M.flatten()[M.flatten()>0]
    denom = rowsums[rowsums>0].prod()
    M = M #* denom
    for i in range(len(m)):        
        if rowsums[i] > 0:
            M[i,:] = M[i,:] / rowsums[i]
    return M, denom

def gcds(*args):
    return reduce(gcd, args)


def is_terminal_state(row):
    return all([col == 0 for col in row])

def solution(m):
    
    n = len(m)
    terminal_states = [i for i, row in enumerate(m) if is_terminal_state(row)]
    if len(terminal_states) == n:
        return [1] + [0] * (n-1) + [1]
    if len(terminal_states) == 1:
        return [1,1]
    if len(terminal_states) == 0:
        return [1]
    
    M,d = normalize_matrix(m)

    inv = np.linalg.inv(np.eye(n, dtype=np.int64) - M)
    parts = inv[0] * d / inv[0,0]
    terminal = parts[terminal_states].round().astype('int')
    
    gcd = gcds(*terminal)
    res = terminal / gcd
    denom = sum(res)
    return res.tolist() + [denom]