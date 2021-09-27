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
from functools import reduce


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
    edges = {
        Edge(i,j)
        for i in range(n) for j in range(i+1, n)
        if has_banana_edge(banana_list[i], banana_list[j])
        }

    graph = Graph(list(range(n)), edges)

    matching = graph.find_maximum_matching()

    # each matching edge covers 2 vertices
    return n - 2* len(matching)

def has_banana_edge(b1, b2):
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

    if b1 == b2:
        ## Loop ended.
        return False

 
    if b1 % 2 != b2 % 2:
        # one is odd, one is even, i.e. b1 + b2 is odd
        # There's no way to split b1+b2 equally, so
        # the loop will be infinite.
        return True    

    # check whether we can cancel out unnecessary prime factors
    d = gcd(b1, b2)
    
    if d > 1:
        # We can simplify the pair
        # --> Cancel and recurse
        b1 //= d
        b2 //= d
        return has_banana_edge(b1, b2)
    
    ## Fully Cancelled 
    # --> perform one update step
    # w.l.o.g.: b1 < b2:
    if b1 > b2:
        b1, b2 = b2, b1
    
    b1, b2 = 2*b1, b2-b1
    return has_banana_edge(b1, b2)


class Edge:
    """Abstraction for an Edge (v,w) of a graph."""
    def __init__(self, v,w, directed = False):
        if not directed:
            v,w = min(v,w), max(v,w)
        self.v = v
        self.w = w
        self.directed = directed

    def other(self, v):
        if v == self.v:
            return self.w
        if v == self.w:
            return self.v
        raise ValueError("inavalid vertex.")

    def __str__(self):
        if not self.directed:
            return str((self.v, self.w))
        else:
            return "(" + str(self.v) + "->" + str(self.w) + ")"

    def __repr__(self):
        	return self.__str__()

    def __hash__(self):
        return hash(repr(self))

    def nodes(self):
        return {self.v, self.w}

    def __getitem__(self, key):
        if key == 0:
            return self.v
        elif key ==1:
            return self.w
        else:
            raise IndexError()


    def __contains__(self, v):
        return v in self.nodes()

    def _directed_equal(self, v, w):
        """checks whether this edge is equal to directed edge (v,w)"""
        return (self.v, self.w) == (other_v, other_w)

    def __eq__(self, other):
        if self.directed:
            return self._directed_equal(other.v, other.w)
        return self.nodes() == other.nodes()

    def undirected(self):
        """Returns an undirected version of self."""
        return Edge(self.v, self.w, directed = False)


class Matching(set):
    """A set of edges such that no vertex appears in multiple
    edges of the matching.
    
    (Note: any matching is embedded in a graph, but we don't need that knowledge for
    this implementation.)
    """
    def __init__(self, edges):
        super(Matching, self).__init__(edges)

    def augment(self, path):
        """
        Augments the matching along the given path.
        In place operation, no return.

        Args:
            path (List[int]) ordered list of vertix indices describing
                an augmenting path. The first and last entry must be free vertices,
                and total length must be even"""
        
        #print("augmenting along " + str(path))
        # update edges in matching
        # making sure each edge is described by lower numbered vertex first
        for i in range(0,len(path), 2):
            # even indices of path
            self.add(Edge(path[i], path[i+1]))
            if i+2 < len(path):
                self.remove(Edge(path[i+1], path[i+2]))

        #print("\tMatching updated:" +  str(self))

class Graph:
    def __init__(self, vertices, edges):
        """input: vertices (list of unique indices) and iterable of edges"""
        self.vertices = set(vertices)
        self.edges = set(edges)

        edges_directed = [e.directed for e in edges]
        if all(edges_directed):
            self.directed = True
        elif not any(edges_directed):
            self.directed = False
        else:
            raise ValueError("got a mix of directed and undirected edges.")

    def neighbors(self, v):
        """returns the (out-)neighborhood of v in the graph"""
        if not self.directed:
            return {e.other(v) for e in self.edges if v in e}
        else:
            # only use edges going out of v
            return {e.other(v) for e in self.edges if e.v == v}

            
    def node_edges(self, v):
        """Set of all edges of vertex v""" 
        return {e for e in self.edges if v in e}


    def find_maximum_matching(self):
        """Finds and returns a maximum cardinality matching on graph.

        Returns:
            matching (List[Set[int,int]]) a maximum cardinality matching
            in the Graph.

        We will use Edmond's Blossom algorithm to find the matching.
        """

        # Start with empty matching
        matching = Matching(set())
        free_vertices = self.vertices

        while len(free_vertices) > 1:
            path = self.find_augmenting_path(matching)
            if path is not None:
                matching.augment(path)
            else:
                # no more improvements possible
                break
        return matching

    def get_exposed_vertices(self, matching):
        """Given a matching (set of edges), returns the exposed vertices in
        the graph, i.e. those not covered by the matching."""
        covered_vertices = {v for e in matching for v in e}
        return self.vertices.difference(covered_vertices)

    def find_augmenting_path(self, matching):
        """Finds a path along which the matching can be augmented
        to increase its cardinality.

        To do so, we'll use Edmond's Blossom algorithm. This implementation
        is based on the pseudo-code from wikipedia, i.e. https://en.wikipedia.org/wiki/Blossom_algorithm
        
        Args:
            matching: Set[Tuple[int]]: current matching
        """

        free_vertices = self.get_exposed_vertices(matching)
        if len(free_vertices) < 2:
            # no more augmentation possible
            return None
            
        marked_vertices = set()
        marked_edges = copy(matching)

        #print(f"Looking for a new augmented path. Matching {matching} free nodes {free_vertices}")

        forest = Forest()
        for fv in free_vertices:
            # create tree with leave {fv}
            forest.add_root(fv)

        unmarked_even_vertices = {
            v for v in self.vertices
                .difference(marked_vertices)   # unmarked vertex
                .intersection(forest.vertices) # in forest
            if not forest.distances[v] % 2     # with even distance to its root
            }

        #print(f"\t unmarked even vertices: {unmarked_even_vertices}")

        while unmarked_even_vertices:
            v = unmarked_even_vertices.pop()
            #print("\t Trying Vertex: " + str(v) + " with root " + str(forest.roots[v]))
            
            unmarked_edges = {e for e in self.edges.difference(marked_edges) if v in e}
            #print(f"\t\tumarked edges: {unmarked_edges}")
            while unmarked_edges:
                e = unmarked_edges.pop()
                w = e.other(v)
                #print(f"\t\tTrying edge {e}")

                if not w in forest.vertices:
                    
                    # w is matched, add e and matched edge to F
                    matched_edge = [e for e in matching if w in e][0]
                    x = matched_edge.other(w)
                    forest.add_edge(v,w)
                    forest.add_edge(w,x)
                    if x not in marked_vertices:
                        unmarked_even_vertices.add(x)
                    #print(f"\t\t\t{w} is matched, adding {e} and {Edge(w,x)}")
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
                            #print(f"\t\t\t {w} is in different, tree. Augmenting along {path}.")
                            return path
                        else:
                            # v and w are in same tree
                            ## We found a blossom! --> contract it, look for path in contracted graph
                            blossom = Blossom(self, matching, v, w, forest) 
                            #print(f'\t\t\t {w} is in same tree. Found a blossom... {blossom}')
                            c_graph, c_matching = blossom.contract()
                            #print(f'\t\t\t{c_graph, c_matching}')
                            c_path = c_graph.find_augmenting_path(c_matching)
                            if c_path is not None:
                                return blossom.lift_path(c_path)
                            
                            # this should be unreachable
                            #print("No augmenting path found in contracted graph. Continuing...")

                    #print(f"\t\t\t{w} is in the tree but has odd distance. Skipping.")
                marked_edges.add(e)
            marked_vertices.add(v)
        
        #print(f"No more vertices. No augmenting path found.")
        # no augmenting path exists
        return None


class Blossom:
    def __init__(self, parent_graph, matching, v, w, forest):
        """"
        Initializes a new Blossom in parent_graph found via root v and other
        vertex w in `forest` expansion.

        parent_graph: Graph (the graph that the blossom is embedded in)
        v: the root of the blossom
        w: other vertex in blossom, found via forest expansion
        forest: the forest through which v,w have been found
        matching: the matching determining which edges of Blossom are coverd
        """
        self.parent_graph = parent_graph
        self.matching = matching # matching in parent_graph

        self.root = v
        # blossom is formed by (v,w) and path (v --> common_root_in_forest(v,w) --> w)
        self.edges = set([Edge(v,w)])
        # to get the second path, we take both path's to the root,
        # then cut off the (shared) stem until the tree branches:
        edges_v = forest.get_root_path_edges(v)
        edges_w = forest.get_root_path_edges(w)
        v_to_w = edges_v.symmetric_difference(edges_w)
        self.edges.update(v_to_w)



        self.vertices = {e.v for e in self.edges}
        self.vertices.update({e.w for e in self.edges})


        # edges that go in/out of the blossom
        self.neighborhood_edges = {
            e
            for e in self.parent_graph.edges
            if (e.v in self.vertices) != (e.w in self.vertices)
        }

        self.contracted_index = None

    def path_to_root(w):
        """Returns:
            path: a list of vertices  odd length, starting with a matched
            edge from w, ending in root
        """
        path = []
        while w != self.root:
            path.add(w)
            # w is in two blossom edges, a matched one and an unmatched one
            w_edges = {e for e in blossom if w in e}
            # NOTE: length refers to vertices, not edges!
            # odd number of vertices --> even number of edges!
            if len(path) % 2:
                # length is odd --> next edge is matched
                e = w_edges.intersection(matching).pop()
            else:
                # even --> next edge is unmatched
                e = w_edges.difference(matching).pop()
            # go to next vertex
            w = e.other(w)
        #finally add the root
        path.add(w)
        return path

    def contract(self):
        """Returns contracted graph, and matching,
        where the blossom is replaced by a single node, as well as information
        necessary to undo the operation.
        """
        # size of original graph
        n = len(self.parent_graph.vertices)
        # choose a new unique index of blossom node
        self.contracted_index = max(self.parent_graph.vertices) + 1
        c_vertices = self.parent_graph.vertices \
                           .difference(self.vertices) \
                           .union({self.contracted_index})

        n_c = len(c_vertices)

        # make edges in contracted graph
        # replace all "neighborhood" edges of the blossom,
        # remove all "internal" edges of the blossom
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

        # matching in contracted graph, is original, minus matched edges
        # in blossom. A matched edge into the root may still need to be replaced.
        c_matching = self.matching.difference(self.edges)
        for e in c_matching:
            if self.root in e:
                c_matching.remove(e)
                c_matching.add(Edge(self.root, e.other(self.root)))

        return c_graph, c_matching

    def lift_path(self, c_path):
        """
        Given an augmented path c_path in the contraction c_graph of this blossom,
        return the lifted augmented path in the blossom's parent_graph.
        """
        if c_path is None:
            # Case 0: no augmenting path in contracted graph -->
            #         none in original either
            path =  None

        if self.root not in c_path or \
                self.root == c_path[0] or self.root == c_path[-1]:
            # Case 1: Blossom not in c_path or an (unmatched) leave of it
            # c_path is [x, ..., y] or [x, ..., b] or [b, ... y]
            # we simply replace b by its root v
            # We simply replace the blossom by its root (if at all)
            path = [v if v != self.contracted_index else self.root for v in c_path]
        else:
            # Case 2: path goes 'through' the contracted blossom
            # c_path is [x, ... b, ... y]
            # with connections (x' --> b --> y')
            # then one of these edges contains x, the other contains a different vertex w in b
            # we follow the blossom along the __matched__ edge of w until we reach v.
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
                    # one of these must be root
                    if in_edge.other(in_neighbor) == self.root:
                        entering_through_root = True
                        w = out_edge.other(out_neighbor)
                    else:
                        #exiting through route
                        entering_through_root = False
                        w = in_edge.other(in_neighbor)
                    path_in_blossom = blossom.path_to_root(w)
                    if entering_through_root:
                        path_in_blossom.reverse()
                    path += path_in_blossom

        #print(f"lifted path: {path}")
        return path
    




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
        return {Edge(rp[i], rp[i+1]) for i in range(self.distances[vertex])}








if __name__ == "__main__":
    banana_list = [1,2,3,4]
    test1 = [1,1]
    # this one has a blossom
    test2 = [1, 7, 3, 21, 13, 19]
    banana_list = test2
    print(solution(banana_list))