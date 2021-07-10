
Maxflow package
===============

.. automodule:: maxflow

The module :py:mod:`maxflow` has the classes :py:class:`maxflow.GraphInt` and
:py:class:`maxflow.GraphFloat`. Both have the same methods and behavior. They
only differ in the data type of the flow network. For conciseness, this page
includes the documentation of just one of these classes.

Note that these classes can be accessed with a template-like syntax using the
dictionary ``maxflow.Graph``: ``maxflow.Graph[int]`` and
``maxflow.Graph[float]``.

.. autoclass:: GraphInt
   :members: __init__, add_nodes, add_grid_nodes, add_edge, add_tedge, add_grid_edges, add_grid_tedges, get_node_count, get_edge_count, maxflow, get_segment, get_grid_segments, get_nx_graph

.. automodule:: maxflow.fastmin
   :members:
