
import numpy as np
from numpy.testing import assert_array_equal

from imageio.v3 import imread

import maxflow


def test_simple():

    for dtype in [int, float]:
        g = maxflow.Graph[dtype](2, 2)
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
