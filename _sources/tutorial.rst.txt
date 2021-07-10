
.. _tutorial:

Tutorial
========

.. The *maximum flow* (maxflow) problem is a common technique
   in optimization and graph theory. Given a directed graph where each edge has
   a capacity, i.e., a flow network, the maximum flow problem consists on
   finding a feasible flow between a single source node and a single sink node
   that is maximum.

This tutorial shows some basic examples on how to import and use *PyMaxflow*. It
is aimed to people who are already familiar with the maxflow problem and its
applications in computer vision and image processing, and want to learn the
basic usage of *PyMaxflow*. This is not, in any case, a tutorial on graph-cuts.

Getting started
---------------

Install *PyMaxflow* using pip::

  pip install PyMaxflow

Once installed, import it as usual::

  import maxflow

  # Print the version
  print(maxflow.__version__)

A flow network with two nodes
-----------------------------

This example builds a simple flow network and finds its maximum flow.

.. image:: _static/graph.png
   :scale: 50 %

This network has two *terminal* nodes, the source :math:`s` and the sink
:math:`t`, and two *non-terminal* nodes, labeled 0 and 1. In *PyMaxflow*,
terminal nodes :math:`s` and :math:`t` are always implicitly present in the
network, and it is not necessary (or even possible) to declare them explicitly.
In addition, terminal edges (connecting non-terminal nodes with terminal nodes),
and non-terminal edges (connecting non-terminal nodes), are treated differently.

The following code uses the standard `single-edge` methods of PyMaxflow to build
this simple network. Note that these methods might be slow in practice for
networks with many nodes and edges::

  import maxflow

  # Create a graph with integer capacities, with 2 non-terminal nodes and 2 non-terminal edges.
  # Note that these numbers are just indicative (read below)
  g = maxflow.Graph[int](2, 2)
  # Add two (non-terminal) nodes. Get the index to the first one.
  nodes = g.add_nodes(2)
  # Create the non-terminal edges (forwards and backwards) with the given capacities between nodes 0 and 1.
  g.add_edge(nodes[0], nodes[1], 1, 2)
  # Set the capacities of the terminal edges...
  # ...for the first node
  g.add_tedge(nodes[0], 2, 5)
  # ...for the second node
  g.add_tedge(nodes[1], 9, 4)

The non-terminal edges are created with ``add_edge``. The terminal edges are
created with ``add_tedge``.

The type of the capacities can be *int*, as in the example, or *float*. In that
case, the graph declaration would be::

  g = maxflow.Graph[float](2, 2)

The constructor parameters ``(2, 2)`` are initial estimations of the number of
nodes and the number of non-terminal edges. These estimations do not need to be
correct or even approximate (it is possible to set them to ``0``), but a good
estimation allows for more efficient memory management. Consult the
documentation of the constructor for more details. In this specific example, the
number of nodes and non-terminal edges was known in advance.

Now we can find the maximum flow in the graph::

  flow = g.maxflow()
  print(f"Maximum flow: {flow}")

Finally, we want to know the the partition given by the minimum cut::

  print(f"Segment of the node 0: {g.get_segment(nodes[0])}")
  print(f"Segment of the node 1: {g.get_segment(nodes[1])}")

The method ``get_segment`` returns ``0`` when the given node belongs to the
source partition and ``1`` when the node belongs to the sink partition.

This example is available in :file:`examples/simple.py`. Running the code will
print::

  Maximum flow: 8
  Segment of the node 0: 1
  Segment of the node 1: 0

This means that the minimum cut cuts the graph in this way:

.. image:: _static/graph2.png
   :scale: 50 %

The severed edges are marked with dashed lines. Indeed, the sum of the
capacities of these edges is equal to the maximum flow 8.

Binary image restoration
------------------------

This example shows how to build a 4-connected grid layout of non-terminal nodes
using the advanced multi-edge functions of PyMaxflow. While this example focuses
on a relatively simple 4-connected grid, these multi-edge functions are flexible
to create very complex networks involving many nodes and edges with a few calls.
More details are given in the following section.

We will use the 4-connected grid network to remove Gaussian noise from a binary
image. The original, noise-free image is

.. image:: _static/a.png

The noisy version was obtained adding strong Gaussian noise to the original
image:

.. image:: _static/a2.png

We will restore the image minimizing the energy

.. math::
   E(\mathbf{x}) = \sum_i D_i(x_i) + \sum_{(i,j)\in\mathcal{C}} K|x_i - x_j|.

:math:`\mathbf{x} \in \{0,1\}^N` are the values of the restored image, :math:`N`
is the number of pixels. The unary term :math:`D_i(0)` (resp. :math:`D_i(1)`)
is the penalty for assigning the value 0 (resp. 1) to the i-th pixel. Each
:math:`D_i` depends on the values of the noisy image, which are denoted as
:math:`p_i`:

.. math::
   D_i(x_i) = \begin{cases} p_i & \textrm{if } x_i=0\\ 255-p_i & \textrm{if } x_i=1 \end{cases}.

Thus, :math:`D_i` is low when assigning the label 0 to dark pixels or the label
1 to bright pixels, and high otherwise. The value :math:`K` is the
regularization strength. The larger :math:`K` the smoother the restoration. We
arbitrarily fix it to 50.

The maximum flow algorithm is widely used to minimize energy functions of this
type. We build a network to represent the above energy. This network has a
non-terminal node per image pixel, and the nodes are connected in a 4-connected
grid arrangement. The capacities of all non-terminal edges is :math:`K`. The
capacities of the edges from the source node are set to :math:`D_i(0)`, and the
capacities of the edges to the sink node are :math:`D_i(1)`.

 .. note:: It could be possible to build this network as we did in the first
    example. First, add all the nodes with ``add_nodes``. Then, iterate over the
    nodes adding the non-terminal edges with ``add_edge``, and finally add the
    capacities of the terminal edges calling ``add_tedge`` once per pixel. While
    this approach is feasible, it is very slow in Python, especially when
    dealing with large images or stacks of images.

*PyMaxflow* provides methods for building complex networks with a few calls. The
method ``add_grid_nodes`` adds multiple nodes and returns their indices in a
convenient n-dimensional array with the given shape; ``add_grid_edges`` adds
edges to the grid with a given neighborhood structure (4-connected by default);
and ``add_grid_tedges`` sets the capacities of the terminal edges for multiple
nodes::

  # Create the graph.
  g = maxflow.Graph[int]()
  # Add the nodes. nodeids has the identifiers of the nodes in the grid.
  # Note that nodeids.shape == img.shape
  nodeids = g.add_grid_nodes(img.shape)
  # Add non-terminal edges with the same capacity.
  g.add_grid_edges(nodeids, 50)
  # Add the terminal edges. The image pixels are the capacities
  # of the edges from the source node. The inverted image pixels
  # are the capacities of the edges to the sink node.
  g.add_grid_tedges(nodeids, img, 255-img)

Perform the maxflow computation and get the results::

  # Find the maximum flow.
  g.maxflow()
  # Get the segments of the nodes in the grid.
  # sgm.shape == nodeids.shape
  sgm = g.get_grid_segments(nodeids)

The method ``get_grid_segments`` returns an array with the same shape than
``nodeids``. It is almost equivalent to calling ``get_segment`` once for each
node in ``nodeids``, but much faster, and preserving the shape of the input. For
the i-th cell, the array stores ``False`` if the i-th node belongs to the source
segment (i.e., the corresponding pixel has the label 1) and ``True`` if the node
belongs to the sink segment (i.e., the corresponding pixel has the label 0). We
now get the labels for each pixel::

  # The labels should be 1 where sgm is False and 0 otherwise.
  img2 = np.int_(np.logical_not(sgm))
  # Show the result.
  from matplotlib import pyplot as ppl
  ppl.imshow(img2)
  ppl.show()

The result is:

.. image:: _static/binary.png
   :scale: 75 %

This is a comparison between the original image (left), the noisy one (center)
and the restoration of this example (right):

.. image:: _static/comparison.png
   :scale: 50 %

Complex grids with ``add_grid_edges``
-------------------------------------

The method ``add_grid_edges`` is a powerful tool to create complex network
layouts:

.. image:: _static/layout_01.png
   :scale: 25 %

.. image:: _static/layout_02.png
   :scale: 25 %

.. image:: _static/layout_03.png
   :scale: 25 %

.. image:: _static/layout_04.png
   :scale: 25 %

.. image:: _static/layout_05.png
   :scale: 25 %

.. image:: _static/layout_06.png
   :scale: 25 %

.. image:: _static/layout_07.png
   :scale: 25 %

The best way to understand the potential applications of ``add_grid_edges`` is
to look at the examples of the `PyMaxflow repository
<https://github.com/pmneila/PyMaxflow>`_.

* The file :file:`examples/layout_examples.py` shows a variety of network
  layouts created with ``add_grid_edges``.
* A more advanced example in :file:`examples/layout_example2.py` builds a
  complex layout with several calls to ``add_grid_edges`` and
  ``add_grid_tedges``.
* The file :file:`examples/layout_example3D.py` contains the definition a 3D
  grid layout.

The documentation
of :py:meth:`maxflow.GraphInt.add_grid_edges` also contains a few useful use
cases.

 ..
    comment:: The first argument of ``add_grid_edges`` is ``nodeids``. It contains an array of
    node identifiers with the shape of the grid of nodes where the edges will be
    added.

    .. note:: The ``nodeids`` argument of ``add_grid_edges`` can be reshaped,
        modified, or reorganized to suit the desired layout of the edges to add. It
        can even contain repeated node IDs (for example, to connect a single node to
        an arbitrary set of nodes). It is not necessary in any case that ``nodeids``
        is the output of ``add_grid_nodes``, as happened in the previous section.

    ``add_grid_edges`` determines the edges to add and their capacities using the
    arguments ``weights`` and ``structure``.

    ``weights`` is an array and its shape must be broadcastable to the shape of
    ``nodeids``. Thus, every node will have a associated weight. ``structure`` is an
    array with the same number of dimensions as ``nodeids`` and with an odd shape
    (typically ``structure.shape == (3, 3, ...)``, but this is not necessary).
    ``structure`` defines the local neighborhood of each node.

    Given a node in ``nodeids``, the ``structure`` array is centered on it. Edges
    are created from that node to the nodes of its neighborhood corresponding to
    nonzero entries of ``structure``. The capacity of the new edge will be the
    product of the ``weight`` of the initial node and the corresponding value in
    ``structure``. Additionally, a reverse edge with the same capacity will be added
    if the argument ``symmetric`` is ``True`` (by default).

    Therefore, the ``weights`` argument allows to define an inhomogeneous graph,
    with different capacities in different areas of the grid. ``structure`` defines
    the local neighborhood of the layout and enables anisotropic edges, with
    different capacities depending on their orientation.
