
import pytest

import numpy as np
from numpy.testing import assert_array_equal
from networkx.utils import graphs_equal

from imageio.v3 import imread

import maxflow


@pytest.mark.parametrize("type", [int, float])
def test_simple(type):
    g = maxflow.Graph[type](2, 2)
    nodes = g.add_nodes(2)
    g.add_edge(nodes[0], nodes[1], 1, 2)
    g.add_tedge(nodes[0], 2, 5)
    g.add_tedge(nodes[1], 9, 4)

    # Find the maxflow.
    flow = g.maxflow()

    assert flow == 8
    assert g.get_segment(nodes[0]) == 1
    assert g.get_segment(nodes[1]) == 0


def test_restoration():

    img = imread("examples/a2.png")
    g = maxflow.Graph[int](0, 0)
    nodeids = g.add_grid_nodes(img.shape)
    g.add_grid_edges(nodeids, 50)
    g.add_grid_tedges(nodeids, img, 255-img)

    g.get_nx_graph()

    g.maxflow()
    sgm = g.get_grid_segments(nodeids)

    assert sgm.sum() == 758


@pytest.mark.parametrize("type", [int, float])
def test_copy_empty(type):
    g = maxflow.Graph[type]()
    g2 = g.copy()
    assert g.get_node_count() == g2.get_node_count()
    assert g.get_edge_count() == g2.get_edge_count()


@pytest.mark.parametrize("type", [int, float])
def test_copy(type):
    g = maxflow.Graph[type]()
    nodeids = g.add_grid_nodes((5, 5))
    structure = np.array([[2, 1, 1, 1, 2],
                          [1, 1, 1, 1, 1],
                          [1, 1, 0, 1, 1],
                          [1, 1, 1, 1, 1],
                          [2, 1, 1, 1, 2]])
    g.add_grid_edges(nodeids, 1, structure=structure, symmetric=False)
    g.add_grid_tedges(nodeids, structure, 2-structure)
    g2 = g.copy()

    assert g.get_node_count() == g2.get_node_count()
    assert g.get_edge_count() == g2.get_edge_count()

    nx1 = g.get_nx_graph()
    nx2 = g2.get_nx_graph()

    assert graphs_equal(nx1, nx2)

    g.maxflow()

    nx1_after_mf = g.get_nx_graph()
    nx2_after_mf = g2.get_nx_graph()

    assert not graphs_equal(nx1, nx1_after_mf)
    assert graphs_equal(nx1, nx2_after_mf)


@pytest.mark.parametrize("type", [int, float])
def test_periodic(type):
    g = maxflow.Graph[type]()
    nodeids = g.add_grid_nodes((5, 5))
    g.add_grid_edges(nodeids, 1, periodic=False)
    nx = g.get_nx_graph()
    assert (nodeids[0, 0], nodeids[0, 4]) not in nx.edges
    assert (nodeids[0, 0], nodeids[4, 0]) not in nx.edges
    assert (nodeids[4, 0], nodeids[4, 4]) not in nx.edges
    assert (nodeids[0, 4], nodeids[4, 4]) not in nx.edges
    assert (nodeids[0, 0], nodeids[4, 4]) not in nx.edges
    assert (nodeids[4, 0], nodeids[0, 1]) not in nx.edges

    g = maxflow.Graph[type]()
    nodeids = g.add_grid_nodes((5, 5))
    g.add_grid_edges(nodeids, 1, periodic=True)
    nx = g.get_nx_graph()
    assert (nodeids[0, 0], nodeids[0, 4]) in nx.edges
    assert (nodeids[0, 0], nodeids[4, 0]) in nx.edges
    assert (nodeids[4, 0], nodeids[4, 4]) in nx.edges
    assert (nodeids[0, 4], nodeids[4, 4]) in nx.edges
    assert (nodeids[0, 0], nodeids[4, 4]) not in nx.edges
    assert (nodeids[4, 0], nodeids[0, 1]) not in nx.edges

    g = maxflow.Graph[type]()
    nodeids = g.add_grid_nodes((5, 5))
    g.add_grid_edges(nodeids, 1, periodic=[True, False])
    nx = g.get_nx_graph()
    assert (nodeids[0, 0], nodeids[0, 4]) not in nx.edges
    assert (nodeids[0, 0], nodeids[4, 0]) in nx.edges
    assert (nodeids[4, 0], nodeids[4, 4]) not in nx.edges
    assert (nodeids[0, 4], nodeids[4, 4]) in nx.edges
    assert (nodeids[0, 0], nodeids[4, 4]) not in nx.edges
    assert (nodeids[4, 0], nodeids[0, 1]) not in nx.edges

    g = maxflow.Graph[type]()
    nodeids = g.add_grid_nodes((5, 5))
    structure = np.array([[0, 0, 0],
                          [0, 0, 1],
                          [1, 1, 1]])
    g.add_grid_edges(nodeids, 1, structure=structure, symmetric=True, periodic=True)
    nx = g.get_nx_graph()
    assert (nodeids[0, 0], nodeids[0, 4]) in nx.edges
    assert (nodeids[0, 0], nodeids[4, 0]) in nx.edges
    assert (nodeids[4, 0], nodeids[4, 4]) in nx.edges
    assert (nodeids[0, 4], nodeids[4, 4]) in nx.edges
    assert (nodeids[0, 0], nodeids[4, 4]) in nx.edges
    assert (nodeids[4, 0], nodeids[0, 1]) in nx.edges


def test_aexpansion():

    unary = np.array([
        [
            [5.0, 10.0, 10.0],
            [0.0, 5.0, 10.0],
            [0.0, 0.0, 5.0]
        ],
        [
            [4.0, 5.0, 5.0],
            [5.0, 4.0, 5.0],
            [5.0, 5.0, 4.0]
        ],
        [
            [5.0, 0.0, 0.0],
            [10.0, 5.0, 0.0],
            [10.0, 10.0, 5.0]
        ],
    ]).transpose((1, 2, 0))

    binary = np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0]
    ])

    labels = maxflow.aexpansion_grid(unary, 0.1 * binary)
    result = np.array([
        [1, 2, 2],
        [0, 1, 2],
        [0, 0, 1]
    ])
    assert_array_equal(labels, result)

    labels = maxflow.aexpansion_grid(unary, 2 * binary)
    assert not np.any(labels == 1)


def test_fastmin_edge_cases():

    # Array with 0 spatial dimensions
    unary = np.zeros((0, 0, 3))
    binary = np.ones((3, 3), dtype=np.float64) - np.eye(3, dtype=np.float64)
    labels = maxflow.aexpansion_grid(unary, binary)
    assert labels.shape == (0, 0)

    # Unary term is a scalar
    unary = np.zeros(())
    binary = np.ones(())
    with pytest.raises(ValueError):
        maxflow.aexpansion_grid(unary, binary)
    with pytest.raises(ValueError):
        maxflow.abswap_grid(unary, binary)

    # num_labels is 0
    unary = np.zeros((3, 3, 0))
    binary = np.zeros((0, 0))
    with pytest.raises(ValueError):
        maxflow.aexpansion_grid(unary, binary)
    with pytest.raises(ValueError):
        maxflow.abswap_grid(unary, binary)

    # num_labels mismatch for the unary and the binary terms
    unary = np.zeros((3, 3, 3))
    binary = np.zeros((2, 2))
    with pytest.raises(ValueError):
        maxflow.aexpansion_grid(unary, binary)
    with pytest.raises(ValueError):
        maxflow.abswap_grid(unary, binary)

    # Shape of initial labels do not match the shape of the unary array
    unary = np.zeros((3, 3, 3))
    binary = np.zeros((3, 3))
    labels = np.ones((4, 4), dtype=np.int64)
    with pytest.raises(Exception):
        maxflow.aexpansion_grid(unary, binary, labels=labels)
    with pytest.raises(Exception):
        maxflow.abswap_grid(unary, binary, labels=labels)

    # Initial labels contain values larger than num_labels
    unary = np.zeros((3, 3, 3))
    binary = np.zeros((3, 3))
    labels = np.full((3, 3), 5, dtype=np.int64)
    with pytest.raises(ValueError):
        maxflow.aexpansion_grid(unary, binary, labels=labels)
    with pytest.raises(ValueError):
        maxflow.abswap_grid(unary, binary, labels=labels)

    # Initial labels contain negative values
    unary = np.zeros((3, 3, 3))
    binary = np.zeros((3, 3))
    labels = np.full((3, 3), -1, dtype=np.int64)
    with pytest.raises(ValueError):
        maxflow.aexpansion_grid(unary, binary, labels=labels)
    with pytest.raises(ValueError):
        maxflow.abswap_grid(unary, binary, labels=labels)


def test_abswap():

    unary = np.array([
        [
            [5.0, 10.0, 10.0],
            [0.0, 5.0, 10.0],
            [0.0, 0.0, 5.0]
        ],
        [
            [10.0, 5.0, 5.0],
            [5.0, 10.0, 5.0],
            [5.0, 5.0, 10.0]
        ],
        [
            [5.0, 0.0, 0.0],
            [10.0, 5.0, 0.0],
            [10.0, 10.0, 5.0]
        ],
    ]).transpose((1, 2, 0))

    binary = np.array([
        [0.0, 1.0, 50.0],
        [1.0, 0.0, 1.0],
        [50.0, 1.0, 0.0]
    ])

    labels = maxflow.abswap_grid(unary, binary)

    result = np.array([
        [1, 2, 2],
        [0, 1, 2],
        [0, 0, 1]
    ])

    assert_array_equal(labels, result)
