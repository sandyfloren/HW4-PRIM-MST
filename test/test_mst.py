import pytest
import numpy as np
from mst import Graph
from sklearn.metrics import pairwise_distances


def check_mst(adj_mat: np.ndarray, 
              mst: np.ndarray, 
              expected_weight: int, 
              allowed_error: float = 0.0001):
    """
    
    Helper function to check the correctness of the adjacency matrix encoding an MST.
    Note that because the MST of a graph is not guaranteed to be unique, we cannot 
    simply check for equality against a known MST of a graph. 

    Arguments:
        adj_mat: adjacency matrix of full graph
        mst: adjacency matrix of proposed minimum spanning tree
        expected_weight: weight of the minimum spanning tree of the full graph
        allowed_error: allowed difference between proposed MST weight and `expected_weight`

    TODO: Add additional assertions to ensure the correctness of your MST implementation. For
    example, how many edges should a minimum spanning tree have? Are minimum spanning trees
    always connected? What else can you think of?

    """
    n = mst.shape[0]

    # Check that proposed MST is a spanning tree of G(V, E):
    # H is a spanning tree of G <=> H is connected and has |V| - 1 edges

    # Perform depth-first search to check that graph is connected
    def dfs(G, s, n=None, visited=None):
        if visited is None:
            n = G.shape[0]
            visited = set()
        visited.add(s)
        for v in set(range(0,n)) - visited:
            if v not in visited:
                dfs(G, v, n, visited)
            return visited

    visited = dfs(adj_mat, 0)
    assert len(visited) == n, 'Proposed MST is not connected'
     
    # Check that graph has correct weight and number of edges
    def approx_equal(a, b):
        return abs(a - b) < allowed_error

    total = 0
    num_edges = 0
    for i in range(n):
        for j in range(i+1):
            val = mst[i, j]
            total += val
            if val > 0:
                num_edges += 1
    assert approx_equal(total, expected_weight), 'Proposed MST has incorrect expected weight'
    assert num_edges == n - 1, 'Proposed MST has incorrect number of edges'




def test_mst_small():
    """
    
    Unit test for the construction of a minimum spanning tree on a small graph.
    
    """
    file_path = './data/small.csv'
    g = Graph(file_path)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 8)


def test_mst_single_cell_data():
    """
    
    Unit test for the construction of a minimum spanning tree using single cell
    data, taken from the Slingshot R package.

    https://bioconductor.org/packages/release/bioc/html/slingshot.html

    """
    file_path = './data/slingshot_example.txt'
    coords = np.loadtxt(file_path) # load coordinates of single cells in low-dimensional subspace
    dist_mat = pairwise_distances(coords) # compute pairwise distances to form graph
    g = Graph(dist_mat)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 57.263561605571695)


def test_mst_student():
    """
    TODO: Write at least one unit test for MST construction.
    """
    # Check that MST does not include self-loops
    file_path = './data/slingshot_example.txt'
    coords = np.loadtxt(file_path) # load coordinates of single cells in low-dimensional subspace
    dist_mat = pairwise_distances(coords) # compute pairwise distances to form graph
    
    # Add self loops to each node
    for i in range(dist_mat.shape[0]):
        dist_mat[i, i] = np.random.uniform()
        
    g = Graph(dist_mat)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 57.263561605571695)

