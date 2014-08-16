
Maxflow package
===============

.. automodule:: maxflow

The module :py:mod:`maxflow` has the classes :py:class:`maxflow.GraphInt` and
:py:class:`maxflow.GraphFloat`. Both have the same methods and behavior.
They only differ in the data type with which they work. Therefore,
we only include here the documentation of one of them. You can access these
classes using the dictionary ``maxflow.Graph``: ``maxflow.Graph[int]`` and
``maxflow.Graph[float]``.

.. autoclass:: GraphInt
   :members: __init__, add_nodes, add_grid_nodes, add_edge, add_tedge, add_grid_edges, maxflow, get_segment, get_grid_segments, get_nx_graph

.. automodule:: maxflow.fastmin
   :members:
