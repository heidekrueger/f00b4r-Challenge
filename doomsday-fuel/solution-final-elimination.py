### My third and final solution  attempt (before commenting and refactoring for submission)

### Fraction solution
import math
import numpy as np
from fractions import gcd, Fraction
from functools import reduce
#import sympy as sp

def gauss(A, b):
    ## solves Ax=b with elemantary operations via Gauss elimination
    ## due to only elementary ops, can handle fractions
    ## borrowed and adapted from https://gist.github.com/j9ac9k/6b5cd12aa9d2e5aa861f942b786293b4
    A = np.concatenate([A,b], axis=1)
    n = len(A)
    m = n + 1
    
    for row in range(n):
        pivots = [abs(A[i][row]) for i in range(row, n)]
        i_max = pivots.index(max(pivots)) + row
        
        # Check for singular matrix
        assert A[i_max][row] != 0, "Matrix is singular!"
        
        # Swap rows
        A[row], A[i_max] = A[i_max], A[row]

        
        for i in range(row + 1, n):
            f = A[i][row] / A[row][row]
            for j in range(row + 1, m):
                A[i][j] -= A[row][j] * f

            # Fill lower triangular matrix with zeros:
            A[i][row] = 0
    
    # Solve equation Ax=b for an upper triangular matrix A         
    x = []
    for i in range(n - 1, -1, -1):
        x.insert(0, A[i][n] / A[i][i])
        for row in range(i - 1, -1, -1):
            A[row][n] -= A[row][i] * x[0]
    return x
    
def lcm(a,b):
    return a*b / gcd(a,b)
    
def lcms(*args):
    return reduce(lcm, args)

def normalize_matrix(m):
    M = np.array(
        [[Fraction(e) for e in row] for row in m],
        dtype=Fraction)
    rowsums = M.sum(axis=1)

    for i in range(len(m)):        
        if rowsums[i] > 0:
             M[i,:] = M[i,:] / rowsums[i] 
    return M

def is_terminal_state(row):
    return all([col == 0 for col in row])

def solution(m):
    n = len(m)
    terminal_states = [i for i, row in enumerate(m) if is_terminal_state(row)]
    if len(terminal_states) == n:
        return [1] + [0] * (n-1) + [1]

    M= normalize_matrix(m)
    I = np.zeros([n,n], dtype=Fraction) + np.diag([Fraction(1,1)] * n)
    A = np.transpose(I - M)
    # setup b: initial distribution is [1, 0, 0, ..., 0]
    b = np.full([n, 1], Fraction(0,1))
    b[0,0] = Fraction(1,1)
    
    x = gauss(A,b)
    x = [x[i] for i in terminal_states]
    nums, denoms = zip(*[(e.numerator, e.denominator) for e in x])
    lcm = lcms(*denoms)
    nums = (np.array(nums) * lcm / np.array(denoms))
    return nums.tolist() + [lcm]

    
#input1 = [[0, 2, 1, 0, 0], [0, 0, 0, 3, 4], [0, 0, 0, 0, 0], [0, 0, 0, 0,0], [0, 0, 0, 0, 0]]
#input2 = [[0, 1, 0, 0, 0, 1], [4, 0, 0, 3, 2, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]

#print(solution(input1))