# -*- encoding:utf-8 -*-

import numpy as np
import _maxflow
from _maxflow import Graph, GraphInt, GraphFloat
from version import __version__, __version_str__, \
        __version_core__, __author__, __author_core__

__doc__ = _maxflow.__doc__
SOURCE = _maxflow.termtype.SOURCE
SINK = _maxflow.termtype.SINK

def _add_grid_nodes(self, shape):
    """Create a new grid of nodes with the given shape, and returns
    an array of the same shape containing the identifiers of the new nodes.
    """
    num_nodes = np.prod(shape)
    first = self.add_nodes(int(num_nodes))
    nodes = np.arange(first, first+num_nodes)
    return np.reshape(nodes, shape)

GraphInt.add_grid_nodes = _add_grid_nodes
GraphFloat.add_grid_nodes = _add_grid_nodes
