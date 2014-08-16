
"""
This file contains a list of examples with different layouts that can be
obtained using the ``add_grid_edges`` method.
"""

import numpy as np
import maxflow
import networkx as nx
import matplotlib.pyplot as plt

# Node positions for drawing
def plot_graph(nxgraph):
    X, Y = np.mgrid[:5, :5]
    aux = np.array([Y.ravel(), X[::-1].ravel()]).T
    positions = {i:aux[i] for i in xrange(25)}
    
    nxgraph.remove_nodes_from(['s', 't'])
    plt.clf()
    nx.draw(nxgraph, pos=positions)
    plt.axis('equal')
    plt.show()

# Standard 4-connected grid
g = maxflow.Graph[int]()
nodeids = g.add_grid_nodes((5,5))
g.add_grid_edges(nodeids, 1)
# Equivalent to
# structure = maxflow.vonNeumann_structure(ndim=2, directed=True)
# g.add_grid_edges(nodeids, 1,
#                           structure=structure,
#                           symmetric=True)
nxgraph = g.get_nx_graph()
plot_graph(nxgraph)

# 8-connected grid
g = maxflow.Graph[int]()
nodeids = g.add_grid_nodes((5,5))
structure = np.array([[0,0,0],
                      [0,0,1],
                      [1,1,1]])
# Also structure = maxflow.moore_structure(ndim=2, directed=True)
g.add_grid_edges(nodeids, 1, structure=structure, symmetric=True)
plot_graph(g.get_nx_graph())

# 24-connected 5x5 neighborhood
g = maxflow.Graph[int]()
nodeids = g.add_grid_nodes((5,5))
structure = np.array([[0,0,0,0,0],
                      [0,0,0,0,0],
                      [0,0,0,1,1],
                      [1,1,1,1,1],
                      [1,1,1,1,1]])
g.add_grid_edges(nodeids, 1, structure=structure, symmetric=True)
plot_graph(g.get_nx_graph())

# Diagonal, not symmetric
g = maxflow.Graph[int]()
nodeids = g.add_grid_nodes((5,5))
structure = np.array([[0,0,0],
                      [0,0,0],
                      [0,0,1]])
g.add_grid_edges(nodeids, 1, structure=structure, symmetric=False)
plot_graph(g.get_nx_graph())

# Central node connected to every other node
g = maxflow.Graph[int]()
nodeids = g.add_grid_nodes((5,5)).ravel()

central_node = nodeids[12]
rest_of_nodes = np.hstack([nodeids[:12], nodeids[13:]])

nodeids = np.empty((2, 24), dtype=np.int_)
nodeids[0] = central_node
nodeids[1] = rest_of_nodes

structure = np.array([[0,0,0],
                      [0,0,0],
                      [0,1,0]])
g.add_grid_edges(nodeids, 1, structure=structure, symmetric=False)
plot_graph(g.get_nx_graph())
