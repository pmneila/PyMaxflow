# -*- encoding:utf-8 -*-

"""
maxflow
=======

``maxflow`` is a Python module for max-flow/min-cut computations. It wraps
the C++ maxflow library by Vladimir Kolmogorov, which implements the
algorithm described in

        An Experimental Comparison of Min-Cut/Max-Flow Algorithms for Energy
        Minimization in Vision. Yuri Boykov and Vladimir Kolmogorov. TPAMI.

This module aims to simplify the construction of graphs with complex
layouts. It provides two Graph classes, ``Graph[int]`` and ``Graph[float]``,
for integer and real data types.

Example:

>>> import maxflow
>>> g = maxflow.Graph[int](2, 2)
>>> nodes = g.add_nodes(2)
>>> g.add_edge(nodes[0], nodes[1], 1, 2)
>>> g.add_tedge(nodes[0], 2, 5)
>>> g.add_tedge(nodes[1], 9, 4)
>>> g.maxflow()
8
>>> g.get_grid_segments(nodes)
array([ True, False])

If you use this library for research purposes, you must cite the aforementioned
paper in any resulting publication.
"""

import numpy as np
from . import _maxflow
from ._maxflow import GraphInt, GraphFloat, moore_structure, vonNeumann_structure
from .version import __version__, __version_str__, __version_core__

Graph = {int: GraphInt, float: GraphFloat}

__all__ = ['Graph', "GraphInt", "GraphFloat", "np", "_maxflow",
           "moore_structure", "vonNeumann_structure"]
