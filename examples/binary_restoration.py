
import numpy as np
import scipy
from scipy.misc import imread
from matplotlib import pyplot as ppl

import maxflow

img = imread("a2.png")

# Create the graph.
g = maxflow.Graph[int](0, 0)
# Add the nodes.
nodeids = g.add_grid_nodes(img.shape)
# Add edges with the same capacities.
g.add_grid_edges(nodeids, 50)
# Add the terminal edges.
g.add_grid_tedges(nodeids, img, 255-img)

graph = g.get_nx_graph()

# Find the maximum flow.
g.maxflow()
# Get the segments.
sgm = g.get_grid_segments(nodeids)

# The labels should be 1 where sgm is False and 0 otherwise.
img2 = np.int_(np.logical_not(sgm))
# Show the result.
ppl.imshow(img2, cmap=ppl.cm.gray, interpolation='nearest')
ppl.show()
