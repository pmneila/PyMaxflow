
"""
This file contains a list of examples with different layouts that can be
obtained using the ``add_grid_edges`` method.
"""

import numpy as np
import maxflow
import matplotlib.pyplot as plt

from examples_utils import plot_graph_2d

# Standard 4-connected grid
g = maxflow.Graph[int]()
nodeids = g.add_grid_nodes((5, 5))
g.add_grid_edges(nodeids, 1)
# Equivalent to
# structure = maxflow.vonNeumann_structure(ndim=2, directed=False)
# g.add_grid_edges(nodeids, 1,
#                  structure=structure,
#                  symmetric=False)
plot_graph_2d(g, nodeids.shape, plot_terminals=False)

# 8-connected grid
g = maxflow.Graph[int]()
nodeids = g.add_grid_nodes((5, 5))
structure = np.array([[0, 0, 0],
                      [0, 0, 1],
                      [1, 1, 1]])
# Also structure = maxflow.moore_structure(ndim=2, directed=True)
g.add_grid_edges(nodeids, 1, structure=structure, symmetric=True)
plot_graph_2d(g, nodeids.shape, plot_terminals=False)

# 24-connected 5x5 neighborhood
g = maxflow.Graph[int]()
nodeids = g.add_grid_nodes((5, 5))
structure = np.array([[1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1],
                      [1, 1, 0, 1, 1],
                      [1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1]])
g.add_grid_edges(nodeids, 1, structure=structure, symmetric=False)
plot_graph_2d(g, nodeids.shape, plot_terminals=False, plot_weights=False)

# Diagonal, not symmetric
g = maxflow.Graph[int]()
nodeids = g.add_grid_nodes((5, 5))
structure = np.array([[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 1]])
g.add_grid_edges(nodeids, 1, structure=structure, symmetric=False)
plot_graph_2d(g, nodeids.shape, plot_terminals=False)

# Central node connected to every other node
g = maxflow.Graph[int]()
nodeids = g.add_grid_nodes((5, 5)).ravel()

central_node = nodeids[12]
rest_of_nodes = np.hstack([nodeids[:12], nodeids[13:]])

nodeids = np.empty((2, 24), dtype=np.int_)
nodeids[0] = central_node
nodeids[1] = rest_of_nodes

structure = np.array([[0, 0, 0],
                      [0, 0, 0],
                      [0, 1, 0]])
g.add_grid_edges(nodeids, 1, structure=structure, symmetric=False)
plot_graph_2d(g, (5, 5), plot_terminals=False)
