import maxflow

# Create a graph with integer capacities.
g = maxflow.Graph[int](2, 2)
# Add two (non-terminal) nodes. Get the index to the first one.
nodes = g.add_nodes(2)
# Create two edges (forwards and backwards) with the given capacities.
# The indices of the nodes are always consecutive.
g.add_edge(nodes[0], nodes[1], 1, 2)
# Set the capacities of the terminal edges...
# ...for the first node.
g.add_tedge(nodes[0], 2, 5)
# ...for the second node.
g.add_tedge(nodes[1], 9, 4)

# Find the maxflow.
flow = g.maxflow()
print("Maximum flow: {}".format(flow))

# Print the segment of each node.
print("Segment of the node 0: {}".format(g.get_segment(nodes[0])))
print("Segment of the node 1: {}".format(g.get_segment(nodes[1])))
