
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt

def plot_graph(nxgraph, plot_weights=True, plot_terminals=True):
    X, Y = np.mgrid[:5, :5]
    aux = np.array([Y.ravel(), X[::-1].ravel()]).T
    positions = {i: aux[i] for i in xrange(25)}
    positions['s'] = (-1, 2)
    positions['t'] = (5, 2)
    
    if not plot_terminals:
        nxgraph.remove_nodes_from(['s', 't'])
    
    plt.clf()
    nx.draw(nxgraph, pos=positions)
    
    if plot_weights:
        edge_labels = {}
        for u, v, d in nxgraph.edges(data=True):
            edge_labels[(u,v)] = d['weight']
        nx.draw_networkx_edge_labels(nxgraph, pos=positions, edge_labels=edge_labels, label_pos=0.3, font_size=7)
    
    plt.axis('equal')
    plt.show()
