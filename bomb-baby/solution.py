def solution(m,f):
    """
    Solution to the bomb-baby puzzle.
    
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

    ## Python cannot handle/optimize tail recursion, so 
    ## to avoid hitting the maximum recursion depth, we'll
    ## instead loop until we hit a base case:
    while True:
        # The entire generation poset is symmetric,
        # so we can reduce the problem to m >= f
        # without affecting the output in any way
        if m < f:
            m, f = f, m

        ## Edge cases:    
        # Edge case 1: Negative or zero inputs? --> impossible
        if m<1 or f<1:
            return "impossible"

        # Edge case 2: (m, 1)
        # It takes (m-1) generations to generate the tuple (m, 1), i.e.
        # when always choosing the transition (m,1) -> (m+1, 1) 
        if f==1:
            return str(n + m-1)

        ## Recursive case: go down the tree
        # (m,f) could have been generated from (m-f, f) or (m, f-m)
        # (or their symmetries) but we know that m >= f
        # so f-m will be <1 and we can ignore that branch

        # As long as m will remain greater than f after the update,
        # we already know that would end up in the recursive case again
        # in the next step, so we can directly take multiple steps at
        # once in order to avoid unnecessary iterations:
        steps = m // f

        n+= steps
        m -= steps * f
