
"""
How to use draw_graph3D to create a tridimensional grid connected graph
"""

import numpy as np
import maxflow

from examples_utils import plot_graph_3D


def create_graph(width=6, height=5, depth=2):
    I = np.arange(width*height*depth).reshape(depth, height, width)
    g = maxflow.Graph[float]()
    nodeids = g.add_grid_nodes(I.shape)
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

    X, Y = np.mgrid[:I.shape[0], :I.shape[1]]
    X, Y = X.reshape(1, np.prod(X.shape)), Y.reshape(1, np.prod(Y.shape))
    # Source node connected to leftmost non-terminal nodes.
    left_most = np.concatenate((X, Y, np.zeros_like(X))).astype(np.uint64)
    left_most = np.ravel_multi_index(left_most, I.shape)
    g.add_grid_tedges(left_most, np.inf, 0)
    # Sink node connected to rightmost non-terminal nodes.
    right_most = left_most + I.shape[2] - 1
    g.add_grid_tedges(right_most, 0, np.inf)
    return I, g

if __name__ == '__main__':
    nodeids, g = create_graph()
    plot_graph_3D(g, nodeids.shape)
    g.maxflow()
    print g.get_grid_segments(nodeids)
