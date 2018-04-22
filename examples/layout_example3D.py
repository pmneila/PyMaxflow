
"""
How to use draw_graph3D to create a tridimensional grid connected graph
"""

import numpy as np
import maxflow

from examples_utils import plot_graph_3d

def create_graph(width=6, height=5, depth=2):
    g = maxflow.Graph[float]()
    nodeids = g.add_grid_nodes((depth, height, width))
    structure = np.array([[[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]],
                           [[0, 1, 0],
                            [1, 0, 1],
                            [0, 1, 0]],
                           [[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]]])
    g.add_grid_edges(nodeids, structure=structure)
    
    # Source node connected to leftmost non-terminal nodes.
    g.add_grid_tedges(nodeids[:, :, 0], np.inf, 0)
    # Sink node connected to rightmost non-terminal nodes.
    g.add_grid_tedges(nodeids[:, :, -1], 0, np.inf)
    return nodeids, g

if __name__ == '__main__':
    nodeids, g = create_graph()
    plot_graph_3d(g, nodeids.shape)
    g.maxflow()
    print(g.get_grid_segments(nodeids))
