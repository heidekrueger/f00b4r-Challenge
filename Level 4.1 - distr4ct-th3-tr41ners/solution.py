"""
In this riddle, we need to find a matching between bunny trainers of
maximum cardinality, such that each matched pair plays an infinite 
loop of thumb_wrestling. In each thumb-wrestling round, trainers win/loose a
certain number of bananas - the player with fewer bananas will always double 
their bananas at the cost of the player who starts the match with more bananas.
A match ends when both trainers have the same number of bananas available.

We'll thus consider an unweighted, undirected graph G=(V,E) where the
vertices V are the trainers and (v1,v2) is an edge iff those two trainers could be
matched into an infinite thumb wrestling loop based on their number of 
bananas. In the first step, we will thus determine which pairs of
trainers share an edge.

Unfortunately, the resulting graph is not bipartite. We therefore cannot
use the Hopcroft-Karp algorithm for maximum matchings in bipartite graphs directly,
but will instead rely on the Blossom Algorithm for maximum matchings in general graphs
to find augmented paths. To do so, we'll implemented some graph abstractions providing
us with the necessary functionality. 
"""

from fractions import gcd
from copy import copy

def solution(banana_list):
    """Calculates the minimal number of bunny trainers that can
    not be matched into an infinite loop.

    Args:
        banana_list (List[int]): list of number of bananas per trainer
            (constraints: entries between 1 and 2^30-1, length below 100)
    Output:
        n (int): the minimal number of "unmatched"  trainers.
    """

    n = len(banana_list)
    edges = {Edge(i,j)
             for i in range(n) for j in range(i+1, n)
             if has_infinite_loop(banana_list[i], banana_list[j])
            }
    graph = Graph(list(range(n)), edges)
    matching = graph.find_maximum_matching()
    # each matching edge covers 2 vertices
    return n - 2* len(matching)

def has_infinite_loop(b1, b2):
    """
    For two vertices b1, b2 (represented by their banana-number),
    determine wether there should be an edge (infinite-loop pairing)
    between them.

    Some observations that we will rely on:
    - Symmetry: (n,m) terminates iff (m,n) terminates
    - Simplification/canceling: for any positive integer k:
      (kn, km) terminates iff (n,m) terminates. In that case we can
      simplify the problem to one of smaller total numbers of bananas n+m
    - If n+m is odd, no equal split is possible, so the loop will be infinite.
    - If n+m is even and a thumb wrestling match is performed, then in the
      resulting new state (n', m') both n' and m' will be even. Therefore,
      it will always be possible to cancel the new pair by a factor of 
      at least 2.

    This function thus either returns a result, cancels the pair and recurses
    or performs one thumb-wrestling match and recurses.

    Therefore, for every two recursions (thumb-wrestle, then cancel), we are 
    guaranteed to reduce the problem size by at least one (binary) order of magnitude. 
    
    For inputs below 2**30 (n+m < 2**31), this function is guaranteed to always
    terminate with a recursion depth smaller than 2 * log(2**31) = 62 

    Returns:
        (bool): True theres an edge (ininite loop), False otherwise
    """

    if b1 == b2:
        return False
    if b1 % 2 != b2 % 2:
        return True    

    # check whether we can cancel out unnecessary prime factors
    d = gcd(b1, b2)
    if d > 1:
        # We can simplify the pair --> Cancel and recurse
        b1 //= d
        b2 //= d
        return has_infinite_loop(b1, b2)
    
    ## Fully Cancelled --> perform an iteration of thumb-wrestling
    if b1 > b2: # w.l.o.g.: b1 < b2:
        b1, b2 = b2, b1
    b1, b2 = 2*b1, b2-b1
    return has_infinite_loop(b1, b2)

#########################################
## Graph Abstractions                 ###
#########################################


class Edge(tuple):
    """Abstraction representing an edge (v,w) in a graph."""
    def __new__(self, v,w):
      v,w = min(v,w), max(v,w)
      return tuple.__new__(Edge, (v,w))

    @property
    def v(self):
        """The first (smaller) vertex in e."""
        return self[0]

    @property
    def w(self):
        """The second (larger) vertex in e."""
        return self[1]

    def other(self, v):
        """Returns the other vertex in the e"""
        if v == self.v:
            return self.w
        if v == self.w:
            return self.v
        raise ValueError("Invalid vertex provided!")

class Matching(set):
    """A set of edges such that no vertex appears in multiple
    edges of the matching.
    NOTE: This implementation has no knowledge of the graph and does
        not enforce that a given initialization is indeed a matching.
    """
    def __init__(self, edges):
        super(Matching, self).__init__(edges)

    def augment(self, path):
        """Augments the matching along a given augmenting path,
        thus increasing its cardinality by one.
        
        NOTE: no input validation, we assume the given path is valid.

        In place operation, no return.
        
        Args:
            path (List[int]) ordered list of vertix indices describing
                an augmenting path. The first and last entry must be free vertices,
                and total length must be even.
        """
        for i in range(0,len(path), 2):
            # even indices of path (path starts with unmatched edge)
            self.add(Edge(path[i], path[i+1]))
            if i+2 < len(path):
                self.remove(Edge(path[i+1], path[i+2]))

class Graph:
    """Abstraction of an (unweighted, undirected) graph G = (V,E)
    with provided functionality to find maximum matchings in G.
    """
    def __init__(self, vertices, edges):
        """input: vertices (list of unique indices) and iterable of edges"""
        self.vertices = set(vertices)
        self.edges = set(edges)

    def neighbors(self, v):
        """Returns the neighborhood of v in the graph"""
        return {e.other(v) for e in self.edges if v in e}

    def find_maximum_matching(self):
        """Finds and returns a maximum cardinality matching on graph, 
        via matching augmentation along alternating paths.

        Returns:
            matching (Set[Edge])
        """
        matching = Matching(set())
        free_vertices = self.vertices

        while len(free_vertices) > 1:
            path = self.find_augmenting_path(matching)
            if path is not None:
                matching.augment(path)
            else: # no more improvements possible
                break
        return matching

    def get_exposed_vertices(self, matching):
        """Given a matching, returns the set of exposed vertices in G,
        i.e. those not covered by the matching."""
        covered_vertices = {v for e in matching for v in e}
        return self.vertices.difference(covered_vertices)

    def find_augmenting_path(self, matching):
        """Finds a `matching`-augmenting path in G via Edmond's Blossom algorithm. 
        
        This implementation uses a Forest-Expansion to find blossoms
        (compare pseudocode at https://en.wikipedia.org/wiki/Blossom_algorithm), but
        BFS or DFS would also be possible.
        
        Args:
            matching: Set[Tuple[int]]: current matching

        Returns:
            augmenting_path: List[int] directed list of vertices
                along which the path runs or `None` if no augmenting
                path exists in the graph. 
        """

        free_vertices = self.get_exposed_vertices(matching)
        if len(free_vertices) < 2:
            return None
            
        marked_vertices = set()
        marked_edges = copy(matching)
        forest = Forest()

        for fv in free_vertices:
            forest.add_tree(fv)

        unmarked_even_vertices = {
            v for v in self.vertices
                .difference(marked_vertices)   # unmarked vertex
                .intersection(forest.vertices) # in forest
            if not forest.distances[v] % 2     # with even distance to its root
            }

        while unmarked_even_vertices:
            v = unmarked_even_vertices.pop()
            unmarked_edges = {e for e in self.edges.difference(marked_edges) if v in e}
            while unmarked_edges:
                e = unmarked_edges.pop()
                w = e.other(v)
                if not w in forest.vertices:
                    # w is matched, add e and matched edge to F
                    matched_edge = [e for e in matching if w in e][0]
                    x = matched_edge.other(w)
                    forest.add_edge(v,w)
                    forest.add_edge(w,x)
                    if x not in marked_vertices:
                        unmarked_even_vertices.add(x)
                else: # w is in the forest
                    if not forest.distances[w] % 2:
                        # distance is even --> w is an "out-node"
                        if forest.roots[v] != forest.roots[w]:
                            # v and w are in different trees, we found an augmenting path:
                            # root(v) --> ... --> v --> w --> ... --> root(w)
                            path = forest.get_root_path(v)  # v --> root(v)
                            path.reverse()                  # root(v) --> v
                            path += forest.get_root_path(w) # w --> root(w)
                            return path
                        else:
                            # v and w are in same tree, we found a blossom! 
                            blossom = Blossom(self, matching, v, w, forest) 
                            c_graph, c_matching = blossom.contract()
                            c_path = c_graph.find_augmenting_path(c_matching)
                            return blossom.lift_path(c_path)
                marked_edges.add(e)
            marked_vertices.add(v)
        # out of vertices, no augmenting path exists
        return None

class Forest:
    """Abstraction of a forest of directed(!) trees.
    Each tree is represented by its vertices and directed edges (root,descendent),
    i.e. every vertex in the tree may be the  root of multiple edges but
    descendent of at most one edge. """ 
    def __init__(self):
        self.vertices = set()
        self.edges = set()
        self.roots = {} #dict matching vertex to its root
        self.distances = {}
        self.parents = {}

    def add_tree(self, root):
        """adds a new tree to the forest, starting at root"""
        if root in self.vertices:
            raise ValueError()
        self.vertices.add(root)
        self.distances[root] = 0
        self.roots[root] = root
    
    def add_edge(self, parent, child):
        if parent not in self.vertices or child in self.vertices:
            raise ValueError()
        self.vertices.add(child)
        self.edges.add((parent, child))
        self.distances[child] = self.distances[parent] + 1
        self.roots[child] = self.roots[parent]
        self.parents[child] = parent

    def get_root_path(self, vertex):
        "returns the path from the vertex to its roow"
        root = self.roots[vertex]
        path = [vertex]
        while vertex != root:
            vertex = self.parents[vertex]
            path.append(vertex)
        return path

    def get_root_path_edges(self, vertex):
        "returns set of edges along root path"
        rp = self.get_root_path(vertex)
        return {Edge(rp[i], rp[i+1]) for i in range(self.distances[vertex])}

class Blossom:
    """"A blossom is an odd cycle in a embeded in a parent graph,
    where all but one vertices in the blossom are covered by a matching.
    We will call the uncovered vertex the blossom's root."""
    def __init__(self, parent_graph, matching, v, w, forest):
        """"
        Directly initializes a Blossom in parent_graph found via root v and
        other vertex w found in the `forest` expansion of Edmond's algorithm above.

        Args:
            parent_graph: Graph that the blossom is embedded
            matching: the matching determining which edges of Blossom are coverd
            v: the root of the blossom
            w: other vertex in blossom, found via forest expansion
            forest: the forest through which v,w have been found. We will use this
                to determine the alternating path v --> w
        """
        self.parent_graph = parent_graph
        self.matching = matching # matching in parent_graph
        self.root = v

        # blossom is formed by (v,w) and path (v --> common_root_in_forest(v,w) --> w)
        self.edges = set([Edge(v,w)])
        # to get the second path, we take both path's to their shared root in the forest,
        # then cut off the (shared) stem until the tree branches:
        edges_v = forest.get_root_path_edges(v)
        edges_w = forest.get_root_path_edges(w)
        v_to_w = edges_v.symmetric_difference(edges_w)
        self.edges.update(v_to_w)

        self.vertices = {e.v for e in self.edges}
        self.vertices.update({e.w for e in self.edges})

        # edges that go in/out of the blossom
        self.neighborhood_edges = {e
                                   for e in self.parent_graph.edges
                                   if (e.v in self.vertices) != (e.w in self.vertices)
                                   }

        self.contracted_index = None # will be set when contracting the blossom.

    def path_to_root(self,w):
        """Returns an alternating path through the blossom from w 
        to the root, starting with a matched edge (unless w is the root, in which
        case we will return [w])

        Returns:
            path: list(vertices)
        """
        path = []
        while w != self.root:
            path.append(w)
            # w is in two blossom edges, a matched one and an unmatched one
            w_edges = {e for e in self.edges if w in e}
            # NOTE: length refers to vertices, not edges!
            # odd number of vertices <=> even number of edges
            if len(path) % 2:
                # length is odd --> next edge is matched
                e = w_edges.intersection(self.matching).pop()
            else:
                # even --> next edge is unmatched
                e = w_edges.difference(self.matching).pop()
            # go to next vertex along e
            w = e.other(w)
        # add the root itself
        path.append(w)
        return path

    def contract(self):
        """Contracts the blossom into a single "supervertex" of its parent graph.
        
        Returns:
            c_graph: contracted parent graph of the blossom
            c_matching: contracted matching in c_graph
        """
        # choose a new unique index of blossom node
        self.contracted_index = max(self.parent_graph.vertices) + 1
        c_vertices = self.parent_graph.vertices \
                           .difference(self.vertices) \
                           .union({self.contracted_index})

        ## Make edges in contracted graph
        # replace all "neighborhood" edges of the blossom, remove all "internal" edges
        c_edges = copy(self.parent_graph.edges)
        for e in self.parent_graph.edges:
            if e in self.edges:
                # internal edge of blossom
                c_edges.remove(e)
            if e in self.neighborhood_edges:
                # replace e
                c_edges.remove(e)
                if e.v in self.vertices:
                    c_edges.add(Edge(self.contracted_index, e.w))
                else:
                    c_edges.add(Edge(self.contracted_index, e.v))
        c_graph = Graph(c_vertices, c_edges)

        # matching in contracted graph is original minus matched edges within
        # plus possibly an edge into the blossom's root
        c_matching = self.matching.difference(self.edges)
        for e in c_matching:
            if self.root in e:
                # if there's a matched edge into the blossom's root, we still
                # replace by a matched edge into the blossom.
                c_matching.remove(e)
                c_matching.add(Edge(self.root, e.other(self.root)))
        return c_graph, c_matching

    def lift_path(self, c_path):
        """
        Given an augmented path c_path in the contraction c_graph of this blossom,
        return the lifted augmented path in the blossom's parent_graph.
        """
        if c_path is None:
            # No augmenting path in contracted graph --> none in original either
            return None
        if self.contracted_index not in c_path:
            # blossom not part of path --> no change
            return c_path
        if self.contracted_index in [c_path[0], c_path[-1]]:
            # Blossom is beginning or end of the path. W.l.o.g. assume it's at the end:
            if self.contracted_index == c_path[0]:
                c_path.reverse()
            # start by adding all non-blossom nodes
            path = c_path[:-1]
            # add the blossom: choose any edge that goes into it from 
            # previous node in path
            in_edge = {e for e in self.neighborhood_edges if c_path[-2] in e}.pop()
            w = in_edge.other(c_path[-2]) # 'first' node in edge
            path += self.path_to_root(w)
            return path

        # Final Case: path goes 'through' the contracted blossom
        path = []
        for i,v in enumerate(c_path):
            if v != self.contracted_index:
                path.append(v)
            else:
                # going through blossom!
                in_neighbor = c_path[i-1]
                out_neighbor = c_path[i+1]
                
                in_edge  = {e for e in self.neighborhood_edges if in_neighbor  in e}.pop()
                out_edge = {e for e in self.neighborhood_edges if out_neighbor in e}.pop()
                # one of these must be root, determine if entering or exiting through it
                if in_edge.other(in_neighbor) == self.root:
                    entering_through_root = True
                    w = out_edge.other(out_neighbor)
                else:
                    entering_through_root = False
                    w = in_edge.other(in_neighbor)
                path_in_blossom = blossom.path_to_root(w)
                if entering_through_root:
                    path_in_blossom.reverse()
                path += path_in_blossom        
        return path
    

if __name__ == "__main__":

    #### Test Cases provided by foo bar
    # banana_list = [1,2,3,4]
    # test1 = [1,1]
    # # this one has a blossom
    # test2 = [1, 7, 3, 21, 13, 19]
    # banana_list = test2
    # print(solution(banana_list))

    ## Large Graph Example from TUM
    # https://algorithms.discrete.ma.tum.de/graph-algorithms/matchings-blossom-algorithm/index_en.html
    V = list(range(16))
    E = {Edge(i,j) for (i,j) in [
        (15,1), (1,0), (0,13), (13,2), (2,3),
        (1,4), (4,5), (4,3), (3,5), (2,6),
        (6,7), (7,14), (14,9), (9,10), (10,12),
        (11,12), (11,8), (12,8), (8,7)                           
        ]}

    g = Graph(V,E)
    print(g.find_maximum_matching())