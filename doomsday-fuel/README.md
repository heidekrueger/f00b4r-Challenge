# Doomsday Fuel

This directory contains my solution attempts at the doomsday fuel challenge.

## First, the math

We're given a square $n \times n$ state transition count matrix $A$, a fixed initial condition $v = \pmatrix{1, 0, \dots, 0}^T$. In the corresponding Markov-Chain, some states are terminal (marked by all-zero rows in $A$).
The goal of the challenge is finding the probability distribtuion of terminal states given the initial state **as rational numbers**.

The challenge is that some states of the chain are periodic (but not the entire chain is!), so we can't just simulate transitions until everything is converged. However, at every time step, some share of the "population" will reach terminal states.

In fact, we want to calculate (terminal parts of) the asymptotic state distribution $x$, where
$$x = \sum_{t=0}^\infty  v^T A^t = v^T \sum_{t=0}^\infty A^t$$

We can rewrite $A$ as a stochastic matrix by normalizing it's rows to 1. That way, all EVs of $A$ will be between 0 and 1 and we can apply the geometric power series identity:

$$ x = v^T \sum_{t=0}^\infty A^t = v^T (I - A)^{-1}$$ and thus we can solve this linear system for $x$, then simply looks at the terminal states to determine the population share.

## Implementations -- my approaches

The challenge here is solving this system in a way that retains the rational nature of the solution, and I (in fact) needed three tries to solve the challenge:

### First try -- the "canonical way"
I started by simply implementing everything in numpy, trying to remain in "integer-world", by keeping track of common demoninators in a separate variable and using some helpful identities to keep numbers integer and small enough. For example, one can easily calculate the least common multiple $d$ of denominators of the fractional entries of the (stochastic) matrix A and rewrite the linear equation as

$$x = dv^T(dI - dA)^{-1}$$

However, at some point one will either have to solve the system or invert a matrix -- neither of which is possible out of the box in numpy in a rational way.

I first tried using real-valued inversions/solves and then a bit of dirty hacking to get the fractional values back (including increasing d to retain as many integers as possible, changing to longdoubles, etc), but to no avail:
My solution passed 9 out of the 10 test cases, in one case I was not able to minimize the inaccuracy of floating point operations enough to retain the original rationals.

This solution attempt is given in `solution-float-hack.py`.

### Sympy

Next, I looked into sympy in order to solve the entire thing exactly. This worked well, however afterwards I discovered that `sympy` wasn't allowed in the challenge's python 2.7 sandbox :(
The code is given in `solution-sympy.py`.

### I guess I'll have to do it myself then...

So I was back to my original integer numpy arrays and needed to solve the system rationally. While none of the (allowed) libraries seem to have functionality for this, many algorithms (most naïvely Guassian elimination -- which is not the fastest, but fast enough as we're limited to $10\times 10$ matrices here) can certainly do so. Rather than writing this from scratch, I found a nice and compact implementation (here)[https://gist.github.com/j9ac9k/6b5cd12aa9d2e5aa861f942b786293b4], which I slightly adapted to the problem.

Pairing this with numpy arrays with a `Fractional` data type, I could then get the exact solution, which is presented in `solution-gaussian-elimination.py`.

## Final Words
Another challenge here was writing the solution in python 2.7 where many of the relevant functionalities (gcd, lcn (both math and numpy), native `prod` for collections) weren't available -- or not where I expected them to be.