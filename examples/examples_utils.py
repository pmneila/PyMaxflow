
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

# Draw a 3D graph
# graph: GraphFloat object
# I_shape: shape of matrix I (nodes matrix)
# plot_terminal: plot s and t nodes
# plot_weights: plot edges weights
# font_size: font size of edges weights
def draw_graph3D(graph, I_shape, plot_terminal=True, plot_weights=True, font_size=10):
    w_h = I_shape[1] * I_shape[2]
    X, Y = np.mgrid[:I_shape[1], :I_shape[2]]
    aux = np.array([Y.ravel(), X[::-1].ravel()]).T
    positions = {i: aux[i] for i in xrange(w_h)}

    for i in xrange(1, I_shape[0]):
        for j in xrange(w_h):
            positions[w_h * i + j] = [positions[j][0] + 0.3 * i, positions[j][1] + 0.2 * i]

    positions['s'] = np.array([-1, int(I_shape[1] / 2)])
    positions['t'] = np.array([I_shape[2] + 0.2 * I_shape[0], int(I_shape[1] / 2)])

    nxg = graph.get_nx_graph()
    if not plot_terminal:
        nxg.remove_nodes_from(['s', 't'])

    nx.draw(nxg, pos=positions)
    nx.draw_networkx_labels(nxg, pos=positions)
    if plot_weights:
        edge_labels = dict([((u, v,), d['weight'])
                     for u, v, d in nxg.edges(data=True)])
        nx.draw_networkx_edge_labels(nxg, pos=positions, edge_labels=edge_labels, label_pos=0.3, font_size=font_size)
    plt.axis('equal')
    plt.show()
