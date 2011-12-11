
.. _tutorial:

Tutorial
========

.. The *maximum flow* (maxflow) problem is a common concept
   in optimization and graph theory. Given a directed graph
   where each edge has a capacity, the maximum flow
   problem consists on
   finding a feasible flow between a single source node and
   a single sink node that is maximum.

This tutorial is
aimed to those who know the maximum flow problem
and its applications to computer vision and graphics.
It explains how to use the *PyMaxflow* library
in some key problems, but it assumes that the reader
knows the theoretic grounds behind them.

If the concepts *maxflow*, *mincut* or *Markov Random Field*
are not familiar to you, this tutorial might
be a waste of time. In that case, you may want to read
the `Wikipedia page <http://en.wikipedia.org/wiki/Maximum_flow_problem>`_
on this topic and the tutorial [BOYKOV06]_.

Getting started
---------------

Once you have installed the PyMaxflow library, you can
import it as usual::

  import maxflow
  print maxflow.__version__

A first example
---------------

The first example consists on constructing and finding the maximum
flow of a custom graph:

.. image:: _static/graph.png
   :scale: 50 %

This graph has two *terminal* nodes, the source :math:`s` and the sink :math:`t`,
and two *non-terminal* nodes, labelled 0 and 1. The code for building
this graph is::

  import maxflow
  
  # Create a graph with integer capacities.
  g = maxflow.Graph[int](2, 2)
  # Add two (non-terminal) nodes. Get the index to the first one.
  n0 = g.add_nodes(2)
  # Create two edges (forwards and backwards) with the given capacities.
  # The indices of the nodes are always consecutive.
  g.add_edge(n0, n0 + 1, 1, 2)
  # Set the capacities of the terminal edges...
  # ...for the first node.
  g.add_tedge(n0, 2, 5)
  # ...for the second node.
  g.add_tedge(n0 + 1, 9, 4)

Pretty straightforward, but some details worth mentioning.
First, the data type of the capacities can be *integer*,
as in the example, or *float*. In that case, the
graph construction should be::

  g = maxflow.Graph[float](2, 2)

Second, the constructor parameters are an initial
estimation of the number of nodes and the number
of non-terminal edges. These estimations do not need
to be perfect, not even approximate. But a better
estimation will lead to a better performance in terms
of memory consumption. Please, consult the
documentation of the constructor for more details.
In this example, we exactly know how many nodes
and non-terminal edges the graph has when
we call the constructor.

Third, you do *not* have to create the terminal nodes.
Every graph have
implicitly defined both nodes. Moreover, you cannot create more
terminal nodes. The non-terminal edges (those connecting
two non-terminal nodes) are created with ``add_edge``. The
terminal edges (those connecting a non-terminal node to a
terminal node) are created with ``add_tweights``.

Now we can find the maximum flow in the graph::

  flow = g.maxflow()
  print "Maximum flow:", flow

Finally, we want to know the shape of the partition
given by the minimum cut::

  print "Segment of the node 0:", g.get_segment(n0)
  print "Segment of the node 1:", g.get_segment(n0 + 1)

The method ``get_segment`` returns ``maxflow.SOURCE`` when the
given node belongs to the partition of the source node (i.e., the
minimum cut severs the terminal edge from the node to the sink),
or ``maxflow.SINK`` otherwise (i.e., the minimum cut severs
the terminal edge from the source to the node).

This example is available in :file:`examples/simple.py`. If you
run this code, you it will print::

  Maximum flow: 8
  Segment of the node 0: SINK
  Segment of the node 1: SOURCE

This means that the minimum cut severs the graph in this way:

.. image:: _static/graph2.png
   :scale: 50 %

The severed edges are marked with dashed lines. Indeed, the sum
of the capacities of these edges is equal to the maximum flow 8.

Binary image restoration
------------------------

Now we proceed to a more involved example.
It is known that one of the first applications of the maxflow
problem is the restoration of binary images. 
We take the binary image

.. image:: _static/a.png

and add strong gaussian noise to it:

.. image:: _static/a2.png

You can download this image from this page using the right-click menu
of your browser. You can load it into Python with::

  import numpy as np
  import scipy
  from scipy.misc import imread
  import maxflow
  
  img = imread("a2.png")

We will restore the image minimizing the energy

.. math::
   E(\mathbf{x}) = \sum_i D_i(x_i) + \sum_{(i,j)\in\mathcal{C}} K|x_i - x_j|.

:math:`\mathbf{x} \in \{0,1\}^N` are the labels of the restored image, :math:`N`
is the number of pixels. The unary term :math:`D_i(0)` (resp :math:`D_i(1)`)
is the penalty for assigning the value 0 (resp 1) to the i-th pixel. Each
:math:`D_i` depends on the values of the noisy image, which are denoted as
:math:`p_i`:

.. math::
   D_i(x_i) = \begin{cases} p_i & \textrm{if } x_i=0\\ 255-p_i & \textrm{if } x_i=1 \end{cases}.

Thus, :math:`D_i` is low when assigning the label 0 to dark pixels or the
label 1 to bright pixels, and high otherwise.
The value :math:`K` is the regularization strength. The larger :math:`K`
the smoother the restoration. We fix it to 50.

The maximum flow algorithm is widely used to minimize energy functions of this
type. We build a graph which represents the above energy. This graph has as many
non-terminal nodes as pixels in the image. The nodes are connected in a grid
arrangement, so that the nodes corresponding to neighbor pixels are connected
by a forward and a backward edge. The capacities of all non-terminal edges
is :math:`K`. The capacities of the edges from the source node are set
to :math:`D_i(0)`, and the capacities of the edges to the sink node are :math:`D_i(1)`.

We could build this graph as in the first example. First, we would add all the nodes.
Then, we would iterate over the nodes adding the edges properly. However, this is extremely
slow in Python, especially when dealing with large images or stacks of images.
*PyMaxflow* provides methods for building some complex graphs with a few calls.
In this example we review ``add_grid_nodes``, ``add_grid_edges``,
which add edges with a fixed capacity to the grid,
and ``add_grid_tedges``, which sets
the capacities of the terminal edges for multiple nodes::

  # Create the graph.
  g = maxflow.Graph[int]()
  # Add the nodes. nodeids has the identifiers of the nodes in the grid.
  nodeids = g.add_grid_nodes(img.shape)
  # Add non-terminal edges with the same capacity.
  g.add_grid_edges(nodeids, 50)
  # Add the terminal edges. The image pixels are the capacities
  # of the edges from the source node. The inverted image pixels
  # are the capacities of the edges to the sink node.
  g.add_grid_tedges(nodeids, img, 255-img)

Finally, we perform the maxflow computation and get the results::

  # Find the maximum flow.
  g.maxflow()
  # Get the segments of the nodes in the grid.
  sgm = g.get_grid_segments(nodeids)

The method ``get_grid_segments`` returns an array with
the same shape than ``nodeids``. It is almost equivalent to calling
``get_segment`` once for each node in ``nodeids``, but much faster.
For the i-th cell, the array stores ``False``
if the i-th node belongs to the ``maxflow.SOURCE`` segment (i.e., the
corresponding pixel has the label 1) and ``True`` if the
node belongs to the ``maxflow.SINK`` segment (i.e., the corresponding
pixel has the label 0). We now get the labels for each pixel 
and reshape the result using the shape of the original image::

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

Fast approximate energy minimization
------------------------------------

TO DO.

.. [BOYKOV06] *Graph Cuts in Vision and Graphics: Theories and Applications*.
   Yuri Boykov, Olga Veksler.
