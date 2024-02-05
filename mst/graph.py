import numpy as np
import heapq
from typing import Union, List, Tuple

class Graph:

    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """
    
        Unlike the BFS assignment, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or a path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph.
    
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')

    def construct_mst(self):
        """
    
        TODO: Given `self.adj_mat`, the adjacency matrix of a connected undirected graph, implement Prim's 
        algorithm to construct an adjacency matrix encoding the minimum spanning tree of `self.adj_mat`. 
            
        `self.adj_mat` is a 2D numpy array of floats. Note that because we assume our input graph is
        undirected, `self.adj_mat` is symmetric. Row i and column j represents the edge weight between
        vertex i and vertex j. An edge weight of zero indicates that no edge exists. 
        
        This function does not return anything. Instead, store the adjacency matrix representation
        of the minimum spanning tree of `self.adj_mat` in `self.mst`. We highly encourage the
        use of priority queues in your implementation. Refer to the heapq module, particularly the 
        `heapify`, `heappop`, and `heappush` functions.

        """
        self.mst = self._prim()

    def _prim(self) -> np.ndarray:
        """
        Compute the MST given lists of nodes, `V`, edges, `E`, and edge weights, `c`. 
        The MST is represented as an adjacency matrix.
        
        Returns:
            a 2D numpy array of floats
        """
        # n: number of nodes in graph
        # S: set of nodes currently in MST
        # S_comp: set of nodes currently not in MST 
        # T: adjacency matrix of MST
        # pi: dictionary of {v: c} pairs s.t. c is the lowest known edge weight between a node u in S and v
        # pred: dictionary of {v: u} pairs where e = (u, v) is the edge between u in S and v s.t. the cost of e is c  
        # s: source node of MST
        # pq: priority queue of [pi[v], v] pairs where v is some node in the graph
        # entry_dict: dictionary to keep track of pq entries

        n = self.adj_mat.shape[0]
        S = set()
        S_comp = set(range(0, n)) 
        T = np.zeros_like(self.adj_mat) 
        pi = {}
        pred = {}     
        pq = []
        entry_dict = {} 
        
        s = 0 
        pi[s] = 0 
        pred[s] = 0
        heapq.heapify(pq)

        # Build pi and pred dictionaries
        for v in range(1, n): # don't include s
            pi[v] = np.inf
            pred[v] = None
        
        # Build priority queue
        for v in range(0, n): # include s
            entry_dict[v] = [pi[v], v]
            heapq.heappush(pq, entry_dict[v])

        # Build MST
        while pq:
            # Get next node not in S
            pi_u, u = heapq.heappop(pq)
            S.add(u)
            S_comp.remove(u)
            pred_u = pred[u]
            if u != s and pred_u is not None:
                T[u, pred_u] = pi[u]
                T[pred_u, u] = pi[u]

            # Update smallest known weights for all edges starting from u to nodes not in S
            for v in S_comp:
                edge_weight = self.adj_mat[u, v]
                if edge_weight > 0 and edge_weight < pi[v]:
                    # Update priority of node v
                    entry_dict[v][0] = edge_weight 
                    pi[v] = edge_weight
                    pq.sort()
                    pred[v] = u
        return T
    


        

