
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

# Find the maxflow.
flow = g.maxflow()
print "Maximum flow:", flow

# Print the segment of each node.
print "Segment of the node 0:", g.what_segment(n0)
print "Segment of the node 1:", g.what_segment(n0 + 1)
