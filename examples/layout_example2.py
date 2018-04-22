
"""
How to use several calls to ``add_grid_edges`` and ``add_grid_tedges`` to
create a flow network with medium complexity.
"""

import numpy as np
import maxflow

from matplotlib import pyplot as plt

from examples_utils import plot_graph_2d

def create_graph():
    g = maxflow.Graph[float]()
    nodeids = g.add_grid_nodes((5,5))
    
    # Edges pointing backwards (left, left up and left down) with infinite
    # capacity
    structure = np.array([[np.inf, 0, 0],
                          [np.inf, 0, 0],
                          [np.inf, 0, 0]
                         ])
    g.add_grid_edges(nodeids, structure=structure, symmetric=False)
    
    # Set a few arbitrary weights
    weights = np.array([[100, 110, 120, 130, 140]]).T + np.array([0, 2, 4, 6, 8])
    
    # Edges pointing right
    structure = np.zeros((3,3))
    structure[1,2] = 1
    g.add_grid_edges(nodeids, structure=structure, weights=weights, symmetric=False)
    
    # Edges pointing up
    structure = np.zeros((3,3))
    structure[0,1] = 1
    g.add_grid_edges(nodeids, structure=structure, weights=weights+100, symmetric=False)
    
    # Edges pointing down
    structure = np.zeros((3,3))
    structure[2,1] = 1
    g.add_grid_edges(nodeids, structure=structure, weights=weights+200, symmetric=False)
    
    # Source node connected to leftmost non-terminal nodes.
    left = nodeids[:, 0]
    g.add_grid_tedges(left, np.inf, 0)
    # Sink node connected to rightmost non-terminal nodes.
    right = nodeids[:, -1]
    g.add_grid_tedges(right, 0, np.inf)
    
    return nodeids, g

if __name__ == '__main__':
    nodeids, g = create_graph()
    
    plot_graph_2d(g, nodeids.shape)
    
    g.maxflow()
    print(g.get_grid_segments(nodeids))
