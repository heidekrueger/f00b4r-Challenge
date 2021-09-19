"""
In this riddle, we need to find a matching between bunny trainers of
maximum cardinality, such that each matched pair plays an infinite 
loop of .

We'll thus consider an unweighted, undirected graph G=(V,E) where the
vertices V are the trainers and (v1,v2) is an edge iff those two trainers could be
matched into an infinite thumb wrestling loop based on their number of 
bananas.

In a first step, we will set up an adjacency matrix, before then
calculating the maximum matching.

Unfortunately, the resulting graph is not bipartite, for example,
if banana_list = [1,2,3,4], we get the following Graph

    1 ----- 2
      \   / |
        X   |
      /   \ |
    3 ----- 4

We therefore cannot use the Hopcroft-Karp algorithm for maximum
matchings in bipartite graphs directly, but will instead rely
on the Blossom Algorithm for maximum matchings in general graphs.
"""

from fractions import gcd
import numpy as np
from copy import copy


def solution(banana_list):
    """Calculates the minimal number of bunny trainers that can
    not be matched into an infinite loop.

    Args:
        banana_list (List[int]): list of number of bananas per trainer
            (constraints: entries between 1 and 2^30-1, length below 100)
    Output:
        n: the minimal number of "unmatched"  trainers.
    """

    n = len(banana_list)

    graph = build_adjacency_matrix(banana_list)
    matching = find_maximum_matching(graph)

    # each matching edge covers 2 vertices
    return n - 2* len(matching)



def build_adjacency_matrix(banana_list):
    """
    Args: 
        banana_list (List[int] of length n)
            Constraints: n <= 100

    Returns:
        A (np.array(shape=[n,n], dtype=bool): Symmetric Adjacency Matrix

    Note: We'll only use the upper triangular part of the matrix, but as
        the maximum input size is 100 trainers, we will optimize here
        to reduce the memory footprint further.
    """
    n = len(banana_list) #number of trainers
    A = np.zeros([n,n])
    for i in range(n):
        for j in range(i+1, n):
            A[i,j] = has_edge(banana_list[i], banana_list[j])
            A[j,i] = A[i,j]

    return A

def has_edge(v1, v2):
    """
    For two vertices v1, v2 (represented by their banana-number),
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

    This function either returns a result, cancels the pair and recurses
    or performs one thumb-wrestling match and recurses.

    Therefore, for every two recursions (thumb-wrestle, then cancel),
    we can guarantee to reduce the problem size by at least an order of magnitude. 
    
    For inputs below 2**30 (n+m < 2**31), this function is 
    guaranteed to always terminate with a recursion depth smaller than
    2 * log(2**31) = 62 

    Returns:
        (bool): True theres an edge (ininite loop), False otherwise
    """

    if v1 == v2:
        ## Loop ended.
        return False

 
    if v1 % 2 != v2 % 2:
        # one is odd, one is even, i.e. v1 + v2 is odd
        # There's no way to split v1+v2 equally, so
        # the loop will be infinite.
        return True    

    # check whether we can cancel out unnecessary prime factors
    d = gcd(v1, v2)
    
    if d > 1:
        # We can simplify the pair
        # --> Cancel and recurse
        v1 //= d
        v2 //= d
        return has_edge(v1, v2)
    
    ## Fully Cancelled 
    # --> perform one update step
    # w.l.o.g.: v1 < v2:
    if v1 > v2:
        v1, v2 = v2, v1
    
    v1, v2 = 2*v1, v2-v1
    return has_edge(v1, v2)


def find_maximum_matching(graph):
    """Finds and returns a maximum cardinality matching on graph.

    Args:
        graph: adjacency matrix, np bool array of shape [n,n] 
            indicating whether an edge exists between two vertices.

    Returns:
        matching (List[Set[int,int]]) a maximum cardinality matching
        in the Graph.

    We will use Edmond's Blossom algorithm to find the matching.
    """

    # Start with empty matching
    matching = set()
    free_vertices = set(range(len(graph)))

    while len(free_vertices) > 1:
        path = find_augmenting_path(graph, matching)
        if path is not None:
            matching = augment(path, matching)
        else:
            # no more improvements possible
            break
        
    return matching

def edge(v1,v2):
    """Returns the sorted tuple (min(v1,v2), max(v1,v2))
    representing an edge between these vertices."""
    return tuple(sorted([v1,v2])) 

def get_free_vertices(graph, matching):
    """returns the set of vertices in graph not
    coverd by the matching"""
    covered_vertices = {v for e in matching for v in e}
    return set(range(len(graph))).difference(covered_vertices)

def augment(path, matching):
    """Augments `matching` along `path`,
    
    Args:
        path (List[int]) ordered list of vertix indices describing
            an augmenting path. The first and last entry must be free vertices,
            and total length must be even
        matching (Set[List[int]]): current matching that can be augmented by
            path
    """
    print("augmenting along " + str(path))
    # update edges in matching
    # making sure each edge is described by lower numbered vertex first
    for i in range(0,len(path), 2):
        # even indices of path
        matching.add(edge(path[i], path[i+1]))
        if i+2 < len(path):
            matching.remove(edge(path[i+1], path[i+2]))

    print("\tnew matching:" +  str(matching))

    return matching

def node_edges(v, all_edges):
    """Set of all edges of vertex v""" 
    return {e for e in all_edges if e[0]==v or e[1]==v}

class Forest:
    """A tree, represented by its vertices,
    and ordered (!) edges (root,descendent),
    i.e. every vertex in the tree may be the
    root of multiple edges but descendent in at most
    one edge. """ 
    def __init__(self):
        self.vertices = set()
        self.edges = set()
        self.roots = {} #dict matching vertex to its root
        self.distances = {}
        self.parents = {}

    def add_root(self, root):
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
        return {edge(rp[i], rp[i+1]) for i in range(self.distances[vertex])}



def other_vertex(edge, first_vertex):
    if edge[0] == first_vertex:
        return edge[1]
    if edge[1] == first_vertex:
        return edge[0]
    raise ValueError('first_vertex not in edge!')

def find_augmenting_path(graph, matching):
    """Finds a path along which the matching can be augmented
    to increase its cardinality.

    To do so, we'll use Edmond's Blossom algorithm. This implementation
    is based on the pseudo-code from wikipedia, i.e. https://en.wikipedia.org/wiki/Blossom_algorithm
    
    Args:
        graph: adjacency matrix
        matching: Set[Tuple[int]]: current matching
        free_vertices: Set[int]: currently free vertices
    """
    n = len(graph)
    # get all vertices and edges
    all_vertices = set(range(n))

    free_vertices = get_free_vertices(graph, matching)
    if len(free_vertices) < 2:
        return None
    
    all_edges = {edge(i,j) 
                 for i in range(n)
                 for j in range(i+1, n)
                 if graph[i,j]}
    
    marked_vertices = set()
    marked_edges = copy(matching)

    


    print(f"Looking for a new augmented path. Matching {matching} free nodes {free_vertices}")

    forest = Forest()
    for fv in free_vertices:
        # create tree with leave {fv}
        forest.add_root(fv)

    unmarked_even_vertices = {
        v for v in all_vertices
            .difference(marked_vertices)   # unmarked vertex
            .intersection(forest.vertices) # in forest
        if not forest.distances[v] % 2     # with even distance to its root
        }

    print(f"\t unmarked even vertices: {unmarked_even_vertices}")

    while unmarked_even_vertices:
        v = unmarked_even_vertices.pop()
        print("\t Trying Vertex: " + str(v) + " with root " + str(forest.roots[v]))
        
        unmarked_edges = {e for e in all_edges.difference(marked_edges) if v in e}
        print(f"\t\tumarked edges: {unmarked_edges}")
        while unmarked_edges:
            e = unmarked_edges.pop()
            w = other_vertex(e, v)
            print(f"\t\tTrying edge {e}")

            if not w in forest.vertices:
                
                # w is matched, add e and matched edge to F
                matched_edge = [e for e in matching if w in e][0]
                x = other_vertex(matched_edge, w)
                forest.add_edge(v,w)
                forest.add_edge(w,x)
                print(f"\t\t\t{w} is matched, adding {e} and {edge(w,x)}")
            else:
                # w is in the forest
                if not forest.distances[w] % 2:
                    # distance is even --> w is an "out-node"

                    if forest.roots[v] != forest.roots[w]:
                        # v and w are in different trees
                        # found an augmenting path!
                        # root(v) --> ... --> v --> w --> ... --> root(w)
                        path = forest.get_root_path(v) # v --> root(v)
                        path.reverse()
                        path += forest.get_root_path(w) # w --> root(w)
                        print(f"\t\t\t {w} is in different, tree. Augmenting along {path}.")
                        return path
                    else:
                        # same tree
                        ## found a blossom --> contract it, look for path in contracted graph
                        
                        # blossom is formed by (v,w), and path from v to w in F
                        edges_v = forest.get_root_path_edges(v)
                        edges_w = forest.get_root_path_edges(w)
                        v_to_w = edges_v.symmetric_difference(edges_w)
                        blossom = v_to_w
                        blossom.add(edge(v,w))
                        print(f'\t\t\t {w} is in same tree. Found a blossom... {blossom}')

                        c_graph, c_matching, c_map = contract_blossom(graph,blossom, v,  matching)
                        print(f'\t\t\t{c_graph, c_matching, c_map}')
                        c_path = find_augmenting_path(c_graph, c_matching)
                        if c_path is None:
                            print("!!!Subpath is none!!!!")
                        path = lift_path(c_path, c_map, blossom, v, graph)
                        return path
                print(f"\t\t\t{w} is in the tree but has odd distance. Skipping.")
            marked_edges.add(e)
        marked_vertices.add(v)
    
    print(f"No more vertices. No augmenting path found.")
    # no augmenting path exists
    return None

def lift_path(c_path, c_to_o, blossom, blossom_root, graph):

    # first step: reconstruct the vertice indices in c_path to original graph
    path = [c_to_o[v] for v in c_path]
    print(f"lifted path: {path}")
    return path

def contract_blossom(graph, blossom, blossom_root, matching):
    """contracts all nodes in blossom into a single node, with the same outgoing edges as blossom_root"""
    blossom_nodes = {e[0] for e in blossom}
    # size of original graph
    n = len(graph)
    contracted_nodes = sorted(set(range(n))
                              .difference(blossom_nodes)
                              .union({blossom_root}))
    n_c = len(contracted_nodes)
    # mapping contracted id to original id
    c_to_o = {c:o for c,o in enumerate(contracted_nodes)}
    o_to_c = {o:c for c,o in enumerate(contracted_nodes)}

    graph_c = np.array([
        graph[row][contracted_nodes] 
        for row in contracted_nodes])
    matching_c = matching.difference(blossom)
    matching_c = {edge(o_to_c[v1], o_to_c[v2]) for (v1, v2) in matching_c}

    return graph_c, matching_c, c_to_o



if __name__ == "__main__":
    banana_list = [1,2,3,4]
    test1 = [1,1]
    # this one has a blossom
    test2 = [1, 7, 3, 21, 13, 19]
    banana_list = test2
    print(solution(banana_list))