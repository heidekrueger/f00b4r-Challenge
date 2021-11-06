"""Solution to Level 5 - Disorderly Escape.

Let $X = [s]^{h \\times w}$, i.e. the set of `h` by `w` matrices over the
field [s], i.e. with integer elements from `{1, ..., s}`, let's call them
colors. Given positive integers w, h, s, we then need to determine the number of 
equivalency classes on X, where A ~ B iff A can be turned into B by permuting
its rows and/or columns.

This is a very nice problem from the intersection of enumerative combinatorics
and group theory that required quite a bit of digging back into textbooks and notes
from my Master's degree. To solve it, we will rely on the following observations:

Let G be the group of "2-D permutations", defined as the subset of
element permutations of a `h` by `w` matrix that can be written as a composition
of "1d-permutations" of the matrix's rows and columns.
Then the number of equivalency classes is equal to the number of orbits $|X/G|$
under the group action g of G acting on all of X.

With this in mind, the main result we will rely on is the Polya enumeration
theorem: https://en.wikipedia.org/wiki/P%C3%B3lya_enumeration_theorem
Burnside's Lemma states that the number of orbits is equal to the average (over g)
number of matrices in X that are invariant under the elements g of G. 
Intuitively, a matrix is invariant under g iff all matrix elements belonging to
a cycle have the same color. Thus, Polya's theorem as a special case of Burnside's
lemma states that it suffices to know the number of cycles c(g) of each element g of G.

Formally, we have
.. math::
    \left| X / G\right| = \frac{1}{|G|} \sum_g s^{c(g)}

We make the following observations: 
* Row and column (1-d) permutation groups are orthogonal and have h! and w!
  elements, respectively, thus their "cross-product group" G has h!w! elements.
* h!w! is a large number, so we will not want to iterate through all 
  group elements. Instead, let's rewrite 
  .. math::
   \sum_g s^{c(g)} = \sum_{i=1}^wh a_i s^i

  where 1 <= i <= wh is the number of cycles in a 2d-permutation and the
  coefficient a_i is defined as the number of elements in G that have 
  i cycles.
    
We are left with finding the individual coefficients a_i. To do so, we make
2 more observations:
* We can rely on symmetries between all 1-d permutations with the same
  amount and length of cycles. These classes correspond exactly to the integer
  partitions. (--> each partition corresponds to one cycle).
* For each class of these in row and column space, we want to know 
  (a) the number of resulting 2-D permutations and (b) their number of cycles.
  For (a), we can simply multiply the 1-D numbers, for those see 
  `count_permutations` below. For (b), see `n_cycles_2d`. 
"""


from math import factorial
from fractions import gcd

def solution(w, h, s):
    # generate 1-d partitions (up to equvalency classes on cycle lengths)
    c_partitions = list(partitions(w))
    r_partitions = list(partitions(h))
    # determine how often each equivalency class occurs
    # NOTE: this count corresponds to permutations, not the partitions,
    # i.e. it takes into account the number of possible orderings in each cycle
    c_counts = {cycles: count_permutations(cycles) for cycles in c_partitions}
    r_counts = {cycles: count_permutations(cycles) for cycles in r_partitions}
    
    # for each possible 2D-permutation cycle length (1 < ... < w*h),
    # how often does it occur?
    cycle_counts = [0] * (w*h)

    for c_cycle_lengths, c_count in c_counts.items():
        for r_cycle_lengths, r_count in r_counts.items():

            # determine number of cycles in combined 2D-permutation
            n_cycles = n_cycles_2d(c_cycle_lengths, r_cycle_lengths)
            # Add the combined permutations to the tally 2d-permutations
            # with this cycle length
            cycle_counts[n_cycles - 1] += c_count * r_count

    ## With all information, we can now apply Polya's theorem
    total_orbits = 0
    for i in range(w*h):
        total_orbits += cycle_counts[i] * s**(i+1)
    total_orbits //= (factorial(w) * factorial(h))
    return str(total_orbits)


def partitions(n, l=1):
    """Efficient implementation to generate the (unordered) integer partitions of n.

    Borrowed from SO-user skovorodkin. https://stackoverflow.com/a/44209393
    """
    yield (n,)
    for i in range(l, n//2 + 1):
        for p in partitions(n-i, i):
            yield (i,) + p

def count_permutations(cycle_lengths):
    """Returns the number of distinct permutations on a set of size
    n=sum(permutation) that have the given cycle_counts.

    Args: 
        permutation (List[int]): the length of the cycles in the permutation
    Returns:
        the number of distinct permutations (int)
 
    NOTE: an alternative implementation via collections.Counter makes the code
        a bit more elegant, but also much slower.
    """
    # track how many elements of the set have not been allocated to a
    # cycle yet
    remaining = sum(cycle_lengths)
    n_ways = 1

    ## iterate through the cycles and count the ways the next cycle could be chosen.
    # Note that the order of cycles of the same length doesn't matter, so we need
    # to account for this when moving to the next length.
    prev_cycle_length = None
    n_cycles_of_same_length = 0
    
    for l in sorted(cycle_lengths):
        # Number of ways to get the next cycle:
        # choose(remaining, l) [which elements in cylce] 
        # * factorial(l-1)     [order of elements in cycle (l-1 because cycle has no 'beginning')]
        # we can simplify this a bit
        n_ways *= factorial(remaining) // l // factorial(remaining-l)
        remaining -= l
        if l == prev_cycle_length:
            n_cycles_of_same_length += 1
        else: # moved to next cycle length
            # order of prev cycles of same length doesn't matter --> divide by number of orderings
            n_ways //= factorial(n_cycles_of_same_length)
            n_cycles_of_same_length = 1
            prev_cycle_length = l
    # account for final cycle group
    n_ways //= factorial(n_cycles_of_same_length)
    return n_ways

def n_cycles_2d(cycles_r, cycles_c):
    """Returns the number of cycles in the 2D-permutation achieved by
    combining a given row and a given column permutation. This operation only depends
    on the number and length of cycles in those two 1D permutations.

    All cycles in the 2d permutation are contained within the combination of single pair
    of a row cycle and a common cycle. However, the reverse is not ture: A single such 
    pair can yield multiple cycles in the 2d combination. In fact, a pair of cycle lengths
    `n` and `m` will generate gcd(n,m) cycles of length nm/gcd(n,m).
    
    Args:
        cycles_r: Iterable[Int] of cycle lengths in the row permutation
        cycles_c: Iterable[Int] of cycle lengths in the col permutation
    
    Returns:
        n_cycles (int) in the resulting 2d-iteration.
    """
    return sum([gcd(n,m) for n in cycles_c for m in cycles_r])