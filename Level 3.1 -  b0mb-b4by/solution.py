def solution(m,f):
    """
    Solution to the bomb-baby puzzle. Here, we are given some dynamical update rules,
    a pair (m,f) could be succeeded either by (m+f,f) or (m,m+f). Given two numbers 
    (m,f), we are tasked with finding the shortest path (in generations) to generate 
    (m,f) from (1,1), or to determine that the outcome is impossible.

    The 'real challenge' is finding an efficient implementation, in particular, avoiding
    deep recursion stacks and 'skipping' recursion loops that can be determined a-priori to be
    unnecessary.
    
    Args:
        m, f (str): string represenations of integers.
        
    Returns:
        (str): number of generations to generate (m,f) from initial state (1,1) or 'impossible'
        
    Constraints:
        m and f should be string represenations of integers below 10^50.
        
    Implementation: We backpropagate through the generation graph until we hit an 
        edge case for which we can make a definitive decision.
    """

    m = int(m)
    f = int(f)

    # number of accumulated generations
    n = 0 

    ## A tail-recursive implementation with constant memory usage
    ## is easily possible, but unfortunately Python cannot handle/optimize
    ## tail recursion, and is still limited by maximum recursion depth.
    ## We'll thus implement tail-recursion implicitly via a loop:
    while True:
        # The entire generation poset is symmetric,
        # so we can reduce the problem to m >= f
        # without affecting the output in any way
        if m < f:
            m, f = f, m

        ## Base cases:    
        # Base case 1: Negative or zero inputs? --> impossible
        if m<1 or f<1:
            return "impossible"

        # Base case 2: (m, 1)
        # It takes (m-1) generations to generate the tuple (m, 1), i.e.
        # when always choosing the transition (m,1) -> (m+1, 1) 
        if f==1:
            return str(n + m-1)

        ## Recursive case: go down the tree
        # (m,f) could have been generated from (m-f, f) or (m, f-m)
        # (or their symmetries) but we know that m >= f
        # so f-m will be <1 and we can ignore that branch.

        # As long as m will remain greater than f after the update,
        # we already know that would end up in the recursive case again
        # in the next step, so we can directly take multiple steps at
        # once in order to avoid unnecessary iterations:
        steps = m // f

        n += steps
        m -= steps * f
