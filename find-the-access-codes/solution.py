def solution(l):
    """solution to the find-the-access code puzzle
    
    Args:
        l (list[int])
        
    Returns:
        (int) Number of "lucky triples" in l, i.e. 
            triples (ijk) with i<j<k and l[i] | l[j] | l[k]
            
    Implementation:
        To avoid a cubic runtime, we'll use dynamic programming
        to first find the number of "lucky doubles" l[j] | l[k] 
        for all j. This will cost quadratic time and linear memory.
        
        Then, for each i, then the triples will be exactly (ijk)
        where l[i]|l[j] and (jk) is a double. This last check will
        again require quadratic time (i*j) resulting in quadratic
        runtime in total.
    """
    length = len(l)
    
    # number of doubles for each j
    n_doubles = [
        # number of k's after j, where ji | jk
        len([lk for lk in l[j+1 :] if not lk % l[j]])
        for j in range(length)
        ]
    # number of triples for each i
    n_triples = [
        n_doubles[j]                # number of doubles
        for i in range(length)     
        for j in range(i+1, length) # for each j after i
        if not l[j] % l[i]          # where li | jk
        ]
    
    #total number of triples
    return sum(n_triples)