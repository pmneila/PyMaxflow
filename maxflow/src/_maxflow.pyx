
# distutils: language = c++
# cython: embedsignature = True

import numpy as np

# Define PY_ARRAY_UNIQUE_SYMBOL
cdef extern from "pyarray_symbol.h":
    pass

cimport cython
from cython.operator cimport dereference as deref
cimport numpy as np

np.import_array()

from libcpp cimport bool as bool_t
from libc.stdint cimport uintptr_t

cdef extern from "fastmin.h":
    cdef object c_aexpansion "aexpansion"(int, np.ndarray, np.ndarray, np.ndarray) except +
    cdef object c_abswap "abswap"(int, int, np.ndarray, np.ndarray, np.ndarray) except +

def aexpansion_grid_step(int alpha, np.ndarray D, np.ndarray V, np.ndarray labels):
    """
    .. note:: Unless you really need to, you should not call this function.

    Perform an iteration of the alpha-expansion algorithm.
    ``labels`` is a N-dimensional array with shape S=(S_1,...,S_N)
    which holds the labels. The labels should be integer values between
    0 and L-1, where L is the number of labels. ``D`` should be an
    N+1-dimensional array with shape (S_1,...,S_N,L).
    D[p1,...,pn,l] is the unary energy of assigning the label l to the
    variable at the position [p1,...,pn].

    ``V`` should be a two-dimensional array (a matrix) with shape (L,L).
    It encodes the binary term. V[l1,l2] is the energy of assigning the
    labels l1 and l2 to neighbor variables. Both ``D`` and ``V`` must be of
    the same type. ``alpha`` indicates the variable that will be expanded
    in this step.

    This function modifies the ``labels`` array in-place and
    returns a tuple with the graph used for the step and
    the energy of the cut. Note that the energy of the cut **IS** the
    energy of the labeling, and can be used directly as the criterion
    of convergence.
    """
    return c_aexpansion(alpha, D, V, labels)

def abswap_grid_step(int alpha, int beta, np.ndarray D, np.ndarray V, np.ndarray labels):
    """
    .. note:: Unless you really need to, you should not call this function.

    Perform an iteration of the alpha-beta-swap algorithm.
    ``labels`` is a N-dimensional array with shape S=(S_1,...,S_N)
    which holds the labels. The labels should be integer values between
    0 and L-1, where L is the number of labels. ``D`` should be an
    N+1-dimensional array with shape (S_1,...,S_N,L).
    D[p1,...,pn,l] is the unary energy of assigning the label l to the
    variable at the position [p1,...,pn,l].

    ``V`` should be a two-dimensional array (a matrix) with shape (L,L).
    It encodes the binary term. V[l1,l2] is the energy of assigning the
    labels l1 and l2 to neighbor variables. Both ``D`` and ``V`` must be of
    the same type. ``alpha`` and ``beta`` are the variables that can be
    swapped in this step.

    This function modifies the ``labels`` array in-place and
    returns a tuple with the graph used for the step and
    the energy of the cut. Note that the energy of the cut is **NOT**
    the energy of the labeling, and cannot be used directly as the
    criterion of convergence.
    """
    return c_abswap(alpha, beta, D, V, labels)

cdef extern from "core/graph.h":
    cdef cppclass Graph[T,T,T]:
        Graph(int, int)
        Graph(Graph)

        void reset()

        int add_node(int)
        void add_edge(int, int, T, T) except +
        void add_tweights(int, T, T) except +
        void add_grid_edges(np.ndarray, object, object, int, object) except +
        void add_grid_tedges(np.ndarray, object, object) except +

        int get_node_num()
        int get_arc_num()

        T maxflow(bool_t reuse_trees)
        void mark_node(int i)
        void mark_grid_nodes(np.ndarray) except +

        int what_segment(int) except +
        np.ndarray get_grid_segments(np.ndarray) except +

        # Inspection methods
        uintptr_t get_first_arc()
        uintptr_t get_next_arc(uintptr_t a)
        long get_arc_from(uintptr_t a)
        long get_arc_to(uintptr_t a)
        T get_rcap(int a)
        T get_trcap(int node)


cdef public class GraphInt [object PyObject_GraphInt, type GraphInt]:
    cdef Graph[long, long, long]* thisptr
    def __cinit__(self, int est_node_num=0, int est_edge_num=0, GraphInt copy_rhs=None):
        """
        ``est_node_num`` gives an estimate of the maximum number of non-terminal
        nodes that can be added to the graph, while ``est_edge_num`` is an
        estimate of the maximum number of non-terminal edges.

        It is possible to add more nodes to the graph than est_node_num (and
        node_num_max can be zero). However, if the count is exceeded, then the
        internal memory is reallocated (increased by 50\%), which is expensive.
        Also, temporarily the amount of allocated memory would be more than
        twice than needed. Similarly for edges.
        """
        if copy_rhs is not None:
            self.thisptr = new Graph[long, long, long](deref(copy_rhs.thisptr))
        else:
            self.thisptr = new Graph[long, long, long](est_node_num, est_edge_num)
    def __dealloc__(self):
        del self.thisptr
    def copy(self):
        """
        Returns a copy of the current graph, including all nodes, edges, and
        edge capacities.

        Note:
        The capacities of the edges may change during the computation of the
        maximum flow. If the graph is copied after the `maxflow` method has been
        called, the capacities in the new graph will reflect the residual
        capacities. To preserve the original capacities, make a copy of the
        graph before calling the `maxflow` method.
        """
        return GraphInt(0, 0, self)
    def reset(self):
        """Remove all nodes and edges."""
        self.thisptr.reset()
    def add_nodes(self, int num_nodes):
        """
        Add non-terminal node(s) to the graph. By default, one node is
        added. If ``num_nodes``>1, then ``num_nodes`` nodes are inserted. It
        returns the identifiers of the nodes added.

        The source and terminal nodes are included in the graph by default, and
        you must not add them.

        **Important:** see note about the constructor.
        """
        first = self.thisptr.add_node(num_nodes)
        return np.arange(first, first+num_nodes)
    def add_grid_nodes(self, shape):
        """
        Add a grid of non-terminal nodes. Return the identifiers of the added
        nodes in an array with the shape of the grid.
        """
        num_nodes = np.prod(shape)
        first = self.thisptr.add_node(int(num_nodes))
        nodes = np.arange(first, first+num_nodes, dtype=np.int_)
        return np.reshape(nodes, shape)
    def add_edge(self, int i, int j, long capacity, long rcapacity):
        """
        Adds a bidirectional edge between nodes ``i`` and ``j`` with the
        weights ``cap`` and ``rev_cap``.

        To add edges between a non-terminal node and terminal nodes, see
        ``add_tedge``.

        **Important:** see note about the constructor.
        """
        self.thisptr.add_edge(i, j, capacity, rcapacity)

    def add_edges(self, i, j, capacity, rev_capacity):
        """
        Adds bidirectional edges between each pair of nodes ``i`` and ``j``
        with the weights ``capacity`` and ``rev_capacity``. All arguments
        are numpy vectors with the same length.
        """
        self._add_edges(i.astype(np.uint32).flatten(),
                        j.astype(np.uint32).flatten(),
                        capacity.astype(np.int64).flatten(),
                        rev_capacity.astype(np.int64).flatten())
    @cython.boundscheck(False)
    def _add_edges(self,
                  np.ndarray[dtype=np.uint32_t, ndim=1, negative_indices=False] i,
                  np.ndarray[dtype=np.uint32_t, ndim=1, negative_indices=False] j,
                  np.ndarray[dtype=np.int64_t, ndim=1, negative_indices=False] capacity,
                  np.ndarray[dtype=np.int64_t, ndim=1, negative_indices=False] rcapacity):
        """
        Adds bidirectional edges between each pair of nodes ``i`` and ``j``
        with the weights ``capacity`` and ``rev_capacity``. All arguments
        are numpy vectors with the same length.
        """
        if len(i) != len(j) or len(i) != len(capacity) or len(i) != len(rcapacity):
            raise ValueError("All vectors must be the same size")
        cdef:
            size_t n = len(i)
            size_t idx

        for 0 <= idx < n:
            self.thisptr.add_edge(i[idx], j[idx], capacity[idx], rcapacity[idx])

    def add_tedge(self, int i, long cap_source, long cap_sink):
        """
        Add an edge 'SOURCE->i' with capacity ``cap_source`` and another edge
        'i->SINK' with capacity ``cap_sink``. This method can be called multiple
        times for each node. Capacities can be negative.

        **Note:** No internal memory is allocated by this call. The capacities
        of terminal edges are stored in each node.
        """
        self.thisptr.add_tweights(i, cap_source, cap_sink)
    def add_grid_edges(self, np.ndarray nodeids, object weights=1, object structure=None, int symmetric=0, object periodic=False):
        """
        Adds edges to a grid of nodes in a structured manner.

        Parameters
        ----------
        nodeids : ndarray
            An array containing the node identifiers of the grid.
        weights : array-like, optional
            An array containing the capacities of the edges starting at every
            node. The shape of `weights` must be broadcastable to the shape of
            `nodeids`. The default value is 1.
        structure : array-like, optional
            Indicates the neighborhood around each node. Edges will be added
            from the central node to the nodes corresponding to non-zero entries
            in the `structure` array. The capacities of these added edges will
            be computed as the product of the weight from `weights` corresponding
            to the node and the value in `structure`. If `structure` is
            None (default), it is equivalent to
            `vonNeumann_structure(ndim=nodeids.ndim, directed=symmetric)`,
            which creates a 4-connected grid.
        symmetric : bool, optional
            If True, for every edge `i->j`, another edge `j->i` with the same
            capacity will be added. The default value is False.
        periodic : bool or array of bools, optional
            Indicates whether the grid is periodic. If True, nodes in the
            borders of the grid will be neighbors of the nodes in the opposite
            border.

            If an array is given, its length must be equal to `nodeids.ndim`.
            `periodic[d]` indicates whether the boundary of the grid is periodic
            along the dimension `d`.

        Examples
        --------

        Standard 4-connected grid, all capacities set to 1:

        >>> g = maxflow.GraphFloat()
        >>> nodeids = g.add_grid_nodes((250, 250))
        >>> structure = np.array([[0, 0, 0],
                                  [0, 0, 1],
                                  [0, 1, 0]])
        >>> # Or structure = maxflow.vonNeumann_structure(ndim=2, directed=True)
        >>> g.add_grid_edges(nodeids, weights=1, structure=structure,
                symmetric=True)

        ::

            XXX----1--->XXX----1--->XXX
            XXX<---1----XXX<---1----XXX
            | ^         | ^         | ^
            | |         | |         | |
           1| |1       1| |1       1| |1
            | |         | |         | |
            V |         V |         V |
            XXX----1--->XXX----1--->XXX    ...
            XXX<---1----XXX<---1----XXX
            | ^         | ^         | ^
            | |         | |         | |
           1| |1       1| |1       1| |1
            | |         | |         | |
            V |         V |         V |
            XXX----1--->XXX----1--->XXX
            XXX<---1----XXX<---1----XXX

                          .                .
                          .                 .
                          .                  .



        4-connected 3x3 grid, different capacities for different positions,
        not symmetric:

        >>> g = maxflow.GraphFloat()
        >>> nodeids = g.add_grid_nodes((3, 3))
        >>> structure = np.array([[0, 0, 0],
                                  [0, 0, 1],
                                  [0, 1, 0]])
        >>> weights = np.array([[1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9]])
        >>> g.add_grid_edges(nodeids, weights=weights, structure=structure,
                symmetric=False)

        ::

            XXX----1--->XXX----2--->XXX
            XXX         XXX         XXX
            |           |           |
            |           |           |
           1|          2|          3|
            |           |           |
            V           V           V
            XXX----4--->XXX----5--->XXX
            XXX         XXX         XXX
            |           |           |
            |           |           |
           4|          5|          6|
            |           |           |
            V           V           V
            XXX----7--->XXX----8--->XXX
            XXX         XXX         XXX



        4-connected 3x3 grid, different capacities for different positions,
        symmetric:

        >>> g = maxflow.GraphFloat()
        >>> nodeids = g.add_grid_nodes((3, 3))
        >>> structure = np.array([[0, 0, 0],
                                  [0, 0, 1],
                                  [0, 1, 0]])
        >>> weights = np.array([[1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9]])
        >>> g.add_grid_edges(nodeids, weights=weights, structure=structure,
                symmetric=True)

        ::

            XXX----1--->XXX----2--->XXX
            XXX<---1----XXX<---2----XXX
            | ^         | ^         | ^
            | |         | |         | |
           1| |1       2| |2       3| |3
            | |         | |         | |
            V |         V |         V |
            XXX----4--->XXX----5--->XXX
            XXX<---4----XXX<---5----XXX
            | ^         | ^         | ^
            | |         | |         | |
           4| |4       5| |5       6| |6
            | |         | |         | |
            V |         V |         V |
            XXX----7--->XXX----8--->XXX
            XXX<---7----XXX<---8----XXX



        4-connected 3x3 grid, different capacities for different positions,
        undirected structure and not symmetric:

        >>> g = maxflow.GraphFloat()
        >>> nodeids = g.add_grid_nodes((3, 3))
        >>> structure = np.array([[0, 1, 0],
                                  [1, 0, 1],
                                  [0, 1, 0]])
        >>> weights = np.array([[1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9]])
        >>> g.add_grid_edges(nodeids, weights=weights, structure=structure,
                symmetric=False)

        ::

            XXX----1--->XXX----2--->XXX
            XXX<---2----XXX<---3----XXX
            | ^         | ^         | ^
            | |         | |         | |
           1| |4       2| |5       3| |6
            | |         | |         | |
            V |         V |         V |
            XXX----4--->XXX----5--->XXX
            XXX<---5----XXX<---6----XXX
            | ^         | ^         | ^
            | |         | |         | |
           4| |7       5| |8       6| |9
            | |         | |         | |
            V |         V |         V |
            XXX----7--->XXX----8--->XXX
            XXX<---8----XXX<---9----XXX



        4-connected 3x3 grid, different capacities for different orientations,
        not symmetric:

        >>> g = maxflow.GraphFloat()
        >>> nodeids = g.add_grid_nodes((3, 3))
        >>> structure = np.array([[0, 1, 0],
                                  [4, 0, 2],
                                  [0, 3, 0]])
        >>> g.add_grid_edges(nodeids, weights=1, structure=structure,
                symmetric=False)

        ::

            XXX----2--->XXX----2--->XXX
            XXX<---4----XXX<---4----XXX
            | ^         | ^         | ^
            | |         | |         | |
           3| |1       3| |1       3| |1
            | |         | |         | |
            V |         V |         V |
            XXX----2--->XXX----2--->XXX
            XXX<---4----XXX<---4----XXX
            | ^         | ^         | ^
            | |         | |         | |
           3| |1       3| |1       3| |1
            | |         | |         | |
            V |         V |         V |
            XXX----2--->XXX----2--->XXX
            XXX<---4----XXX<---4----XXX

        """
        if structure is None:
            structure = vonNeumann_structure(nodeids.ndim, symmetric)

        self.thisptr.add_grid_edges(nodeids, weights, structure, symmetric, periodic)
    def add_grid_tedges(self, np.ndarray nodeids, sourcecaps, sinkcaps):
        """
        Add terminal edges to a grid of nodes, given their identifiers in
        ``nodeids``. ``sourcecaps`` and ``sinkcaps`` are arrays with the
        capacities of the edges from the source node and to the sink node,
        respectively. The shape of all these arrays must be equal.

        This is equivalent to calling ``add_tedge`` for many nodes, but much faster.
        """
        self.thisptr.add_grid_tedges(nodeids, sourcecaps, sinkcaps)
    def get_node_num(self):
        """
        Returns the number of non-terminal nodes.

        This method is available for backward compatilibity. Use
        ``get_node_count`` instead.
        """
        return self.thisptr.get_node_num()
    def get_edge_num(self):
        """
        Returns the number of non-terminal edges.

        This method is available for backward compatilibity. Use
        ``get_edge_count`` instead.
        """
        return self.thisptr.get_arc_num()
    def get_node_count(self):
        """Returns the number of non-terminal nodes."""
        return self.thisptr.get_node_num()
    def get_edge_count(self):
        """Returns the number of non-terminal edges."""
        return self.thisptr.get_arc_num()
    def maxflow(self, reuse_trees=False):
        """
        Perform the maxflow computation in the graph. Returns the capacity of
        the minimum cut or, equivalently, the maximum flow of the graph.

        If flag reuse_trees is true while calling maxflow(), then search trees
        are reused from previous maxflow computation, as described in

            "Efficiently Solving Dynamic Markov Random Fields Using Graph Cuts."
            Pushmeet Kohli and Philip H.S. Torr
            International Conference on Computer Vision (ICCV), 2005
        """
        return self.thisptr.maxflow(reuse_trees)
    def mark_node(self, i):
        """
        If flag reuse_trees is true while calling maxflow(), then search trees
        are reused from previous maxflow computation.

        In this case before calling maxflow() the user must
        specify which parts of the graph have changed by calling mark_node():
          add_tweights(i),set_trcap(i)    => call mark_node(i)
          add_edge(i,j),set_rcap(a)       => call mark_node(i); mark_node(j)

        This option makes sense only if a small part of the graph is changed.
        The initialization procedure goes only through marked nodes then.
        """
        self.thisptr.mark_node(i)
    def mark_grid_nodes(self, nodeids):
        """
        Mark nodes that have changed. This is equivalent to calling ``mark_node``
        for many nodes.
        """
        self.thisptr.mark_grid_nodes(nodeids)
    def get_segment(self, i):
        """Returns which segment the given node belongs to."""
        return self.thisptr.what_segment(i)
    def get_grid_segments(self, np.ndarray nodeids):
        """
        After the maxflow is computed, this function returns which
        segment the given nodes belong to. The output is a boolean array
        of the same shape than the input array ``nodeids``.

        This is equivalent to calling ``get_segment`` for many nodes, but much faster.
        """
        return self.thisptr.get_grid_segments(nodeids)
    def get_nx_graph(self):
        """
        Build a NetworkX DiGraph with the status of the maxflow network. The
        resulting graph will contain the terminal and non-terminal nodes and
        edges. The attribute ``weight`` of every edge will be set to its
        residual capacity. If the residual capacity of an edge is 0, the edge is
        not included in the DiGraph.

        The residual capacity for an edge is defined as the full capacity of the
        edge (set with the methods ``add_tedge``, ``add_edge`` and their
        corresponding grid equivalents, ``add_grid_tedges`` and
        ``add_grid_edges``) minus the amount of flow passing through it.
        Therefore, the weights of the DiGraph depend on the amount of flow
        passing through the network and, in turn, this depends on whether the
        call to ``get_nx_graph`` is done before or after calling
        ``GraphInt.maxflow``.

        Before calling the ``GraphInt.maxflow``, there is no flow and therefore
        the residual capacity of every edge is equal to its full capacity.

        After calling ``GraphInt.maxflow``, a virtual flow traverses the network
        from the source node to the sink node, and the residual capacities will
        be lower than the full capacities. Note that in this case, since
        ``get_nx_graph`` ignores edges with residual capacity 0, the edges in
        the minimum cut will not be included in the final DiGraph.

        Note that this function is slow and should be used only for debugging
        purposes.

        This method requires the Python NetworkX package.
        """

        import networkx as nx
        g = nx.DiGraph()

        # Add non-terminal nodes
        g.add_nodes_from(range(self.get_node_count()))

        # Add non-terminal edges with capacities
        cdef int num_edges = self.get_edge_count()
        cdef uintptr_t e = self.thisptr.get_first_arc()

        cdef int n1
        cdef int n2
        cdef long w
        for i in range(num_edges):

            n1 = self.thisptr.get_arc_from(e)
            n2 = self.thisptr.get_arc_to(e)
            w = self.thisptr.get_rcap(e)

            if w != 0:
                if g.has_edge(n1, n2):
                    g[n1][n2]['weight'] += w
                else:
                    g.add_edge(n1, n2, weight=w)
            e = self.thisptr.get_next_arc(e)

        # Add terminal nodes
        g.add_nodes_from(['s', 't'])

        # Add terminal edges
        cdef int num_nodes = self.get_node_count()
        cdef long rcap
        cdef int segment
        for i in range(num_nodes):

            segment = self.thisptr.what_segment(i)

            g.nodes[i]['segment'] = segment

            rcap = self.thisptr.get_trcap(i)
            if rcap > 0.0:
                g.add_edge('s', i, weight=rcap)
            elif rcap < 0.0:
                g.add_edge(i, 't', weight=-rcap)

        return g


cdef public class GraphFloat [object PyObject_GraphFloat, type GraphFloat]:
    cdef Graph[double, double, double]* thisptr
    def __cinit__(self, int est_node_num=0, int est_edge_num=0, GraphFloat copy_rhs=None):
        """
        ``est_node_num`` gives an estimate of the maximum number of non-terminal
        nodes that can be added to the graph, while ``est_edge_num`` is an
        estimate of the maximum number of non-terminal edges.

        It is possible to add more nodes to the graph than est_node_num (and
        node_num_max can be zero). However, if the count is exceeded, then the
        internal memory is reallocated (increased by 50\%), which is expensive.
        Also, temporarily the amount of allocated memory would be more than
        twice than needed. Similarly for edges.
        """
        if copy_rhs is not None:
            self.thisptr = new Graph[double, double, double](deref(copy_rhs.thisptr))
        else:
            self.thisptr = new Graph[double, double, double](est_node_num, est_edge_num)
    def __dealloc__(self):
        del self.thisptr
    def copy(self):
        """
        Returns a copy of the current graph, including all nodes, edges, and
        edge capacities.

        Note:
        The capacities of the edges may change during the computation of the
        maximum flow. If the graph is copied after the `maxflow` method has been
        called, the capacities in the new graph will reflect the residual
        capacities. To preserve the original capacities, make a copy of the
        graph before calling the `maxflow` method.
        """
        return GraphFloat(0, 0, self)
    def reset(self):
        """Remove all nodes and edges."""
        self.thisptr.reset()
    def add_nodes(self, int num_nodes):
        """
        Add non-terminal node(s) to the graph. By default, one node is
        added. If ``num_nodes``>1, then ``num_nodes`` nodes are inserted. It
        returns the identifiers of the nodes added.

        The source and terminal nodes are included in the graph by default, and
        you must not add them.

        **Important:** see note about the constructor"""
        first = self.thisptr.add_node(num_nodes)
        return np.arange(first, first+num_nodes)
    def add_grid_nodes(self, shape):
        """
        Add a grid of non-terminal nodes. Return the identifiers of the added
        nodes in an array with the shape of the grid.
        """
        num_nodes = np.prod(shape)
        first = self.thisptr.add_node(int(num_nodes))
        nodes = np.arange(first, first+num_nodes, dtype=np.int_)
        return np.reshape(nodes, shape)
    def add_edge(self, int i, int j, double capacity, double rcapacity):
        """
        Adds a bidirectional edge between nodes ``i`` and ``j`` with the
        weights ``cap`` and ``rev_cap``.

        To add edges between a non-terminal node and terminal nodes, see
        ``add_tedge``.

        **Important:** see note about the constructor.
        """
        self.thisptr.add_edge(i, j, capacity, rcapacity)

    def add_edges(self, i, j, capacity, rev_capacity):
        """
        Adds bidirectional edges between each pair of nodes ``i`` and ``j``
        with the weights ``capacity`` and ``rev_capacity``. All arguments
        are numpy vectors with the same length.
        """
        self._add_edges(i.astype(np.uint32).flatten(),
                        j.astype(np.uint32).flatten(),
                        capacity.astype(np.float64).flatten(),
                        rev_capacity.astype(np.float64).flatten())

    @cython.boundscheck(False)
    def _add_edges(self,
                  np.ndarray[dtype=np.uint32_t, ndim=1, negative_indices=False] i,
                  np.ndarray[dtype=np.uint32_t, ndim=1, negative_indices=False] j,
                  np.ndarray[dtype=np.float64_t, ndim=1, negative_indices=False] capacity,
                  np.ndarray[dtype=np.float64_t, ndim=1, negative_indices=False] rcapacity):
        if len(i) != len(j) or len(i) != len(capacity) or len(i) != len(rcapacity):
            raise ValueError("All vectors must be the same size")
        cdef:
            size_t n = len(i)
            size_t idx

        for 0 <= idx < n:
            self.thisptr.add_edge(i[idx], j[idx], capacity[idx], rcapacity[idx])

    def add_tedge(self, int i, double cap_source, double cap_sink):
        """
        Add an edge 'SOURCE->i' with capacity ``cap_source`` and another edge
        'i->SINK' with capacity ``cap_sink``. This method can be called multiple
        times for each node. Capacities can be negative.

        **Note:** No internal memory is allocated by this call. The capacities
        of terminal edges are stored in each node.
        """
        self.thisptr.add_tweights(i, cap_source, cap_sink)

    def add_grid_edges(self, np.ndarray nodeids, object weights=1, object structure=None, int symmetric=0, object periodic=False):
        """
        Adds edges to a grid of nodes in a structured manner.

        Parameters
        ----------
        nodeids : ndarray
            An array containing the node identifiers of the grid.
        weights : array-like, optional
            An array containing the capacities of the edges starting at every
            node. The shape of `weights` must be broadcastable to the shape of
            `nodeids`. The default value is 1.
        structure : array-like, optional
            Indicates the neighborhood around each node. Edges will be added
            from the central node to the nodes corresponding to non-zero entries
            in the `structure` array. The capacities of these added edges will
            be computed as the product of the weight from `weights` corresponding
            to the node and the value in `structure`. If `structure` is
            None (default), it is equivalent to
            `vonNeumann_structure(ndim=nodeids.ndim, directed=symmetric)`,
            which creates a 4-connected grid.
        symmetric : bool, optional
            If True, for every edge `i->j`, another edge `j->i` with the same
            capacity will be added. The default value is False.
        periodic : bool or array of bools, optional
            Indicates whether the grid is periodic. If True, nodes in the
            borders of the grid will be neighbors of the nodes in the opposite
            border.

            If an array is given, its length must be equal to `nodeids.ndim`.
            `periodic[d]` indicates whether the boundary of the grid is periodic
            along the dimension `d`.

        Examples
        --------

        Standard 4-connected grid, all capacities set to 1:

        >>> g = maxflow.GraphFloat()
        >>> nodeids = g.add_grid_nodes((250, 250))
        >>> structure = np.array([[0, 0, 0],
                                  [0, 0, 1],
                                  [0, 1, 0]])
        >>> # Or structure = maxflow.vonNeumann_structure(ndim=2, directed=True)
        >>> g.add_grid_edges(nodeids, weights=1, structure=structure,
                symmetric=True)


        XXX----1--->XXX----1--->XXX
        XXX<---1----XXX<---1----XXX
        | ^         | ^         | ^
        | |         | |         | |
       1| |1       1| |1       1| |1
        | |         | |         | |
        V |         V |         V |
        XXX----1--->XXX----1--->XXX    ...
        XXX<---1----XXX<---1----XXX
        | ^         | ^         | ^
        | |         | |         | |
       1| |1       1| |1       1| |1
        | |         | |         | |
        V |         V |         V |
        XXX----1--->XXX----1--->XXX
        XXX<---1----XXX<---1----XXX

                      .                .
                      .                 .
                      .                  .



        4-connected 3x3 grid, different capacities for different positions,
        not symmetric:

        >>> g = maxflow.GraphFloat()
        >>> nodeids = g.add_grid_nodes((3, 3))
        >>> structure = np.array([[0, 0, 0],
                                  [0, 0, 1],
                                  [0, 1, 0]])
        >>> weights = np.array([[1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9]])
        >>> g.add_grid_edges(nodeids, weights=weights, structure=structure,
                symmetric=False)


        XXX----1--->XXX----2--->XXX
        XXX         XXX         XXX
        |           |           |
        |           |           |
       1|          2|          3|
        |           |           |
        V           V           V
        XXX----4--->XXX----5--->XXX
        XXX         XXX         XXX
        |           |           |
        |           |           |
       4|          5|          6|
        |           |           |
        V           V           V
        XXX----7--->XXX----8--->XXX
        XXX         XXX         XXX



        4-connected 3x3 grid, different capacities for different positions,
        symmetric:

        >>> g = maxflow.GraphFloat()
        >>> nodeids = g.add_grid_nodes((3, 3))
        >>> structure = np.array([[0, 0, 0],
                                  [0, 0, 1],
                                  [0, 1, 0]])
        >>> weights = np.array([[1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9]])
        >>> g.add_grid_edges(nodeids, weights=weights, structure=structure,
                symmetric=True)


        XXX----1--->XXX----2--->XXX
        XXX<---1----XXX<---2----XXX
        | ^         | ^         | ^
        | |         | |         | |
       1| |1       2| |2       3| |3
        | |         | |         | |
        V |         V |         V |
        XXX----4--->XXX----5--->XXX
        XXX<---4----XXX<---5----XXX
        | ^         | ^         | ^
        | |         | |         | |
       4| |4       5| |5       6| |6
        | |         | |         | |
        V |         V |         V |
        XXX----7--->XXX----8--->XXX
        XXX<---7----XXX<---8----XXX



        4-connected 3x3 grid, different capacities for different positions,
        undirected structure and not symmetric:

        >>> g = maxflow.GraphFloat()
        >>> nodeids = g.add_grid_nodes((3, 3))
        >>> structure = np.array([[0, 1, 0],
                                  [1, 0, 1],
                                  [0, 1, 0]])
        >>> weights = np.array([[1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9]])
        >>> g.add_grid_edges(nodeids, weights=weights, structure=structure,
                symmetric=False)


        XXX----1--->XXX----2--->XXX
        XXX<---2----XXX<---3----XXX
        | ^         | ^         | ^
        | |         | |         | |
       1| |4       2| |5       3| |6
        | |         | |         | |
        V |         V |         V |
        XXX----4--->XXX----5--->XXX
        XXX<---5----XXX<---6----XXX
        | ^         | ^         | ^
        | |         | |         | |
       4| |7       5| |8       6| |9
        | |         | |         | |
        V |         V |         V |
        XXX----7--->XXX----8--->XXX
        XXX<---8----XXX<---9----XXX



        4-connected 3x3 grid, different capacities for different orientations,
        not symmetric:

        >>> g = maxflow.GraphFloat()
        >>> nodeids = g.add_grid_nodes((3, 3))
        >>> structure = np.array([[0, 1, 0],
                                  [4, 0, 2],
                                  [0, 3, 0]])
        >>> g.add_grid_edges(nodeids, weights=1, structure=structure,
                symmetric=False)


        XXX----2--->XXX----2--->XXX
        XXX<---4----XXX<---4----XXX
        | ^         | ^         | ^
        | |         | |         | |
       3| |1       3| |1       3| |1
        | |         | |         | |
        V |         V |         V |
        XXX----2--->XXX----2--->XXX
        XXX<---4----XXX<---4----XXX
        | ^         | ^         | ^
        | |         | |         | |
       3| |1       3| |1       3| |1
        | |         | |         | |
        V |         V |         V |
        XXX----2--->XXX----2--->XXX
        XXX<---4----XXX<---4----XXX

       """
        if structure is None:
            structure = vonNeumann_structure(nodeids.ndim, symmetric)

        self.thisptr.add_grid_edges(nodeids, weights, structure, symmetric, periodic)
    def add_grid_tedges(self, np.ndarray nodeids, sourcecaps, sinkcaps):
        """
        Add terminal edges to a grid of nodes, given their identifiers in
        ``nodeids``. ``sourcecaps`` and ``sinkcaps`` are arrays with the
        capacities of the edges from the source node and to the sink node,
        respectively. The shape of all these arrays must be equal.

        This is equivalent to calling ``add_tedge`` for many nodes, but much faster.
        """
        self.thisptr.add_grid_tedges(nodeids, sourcecaps, sinkcaps)
    def get_node_num(self):
        """
        Returns the number of non-terminal nodes.

        This method is available for backward compatilibity. Use
        ``get_node_count`` instead.
        """
        return self.thisptr.get_node_num()
    def get_edge_num(self):
        """
        Returns the number of non-terminal edges.

        This method is available for backward compatilibity. Use
        ``get_edge_count`` instead.
        """
        return self.thisptr.get_arc_num()
    def get_node_count(self):
        """Returns the number of non-terminal nodes."""
        return self.thisptr.get_node_num()
    def get_edge_count(self):
        """Returns the number of non-terminal edges."""
        return self.thisptr.get_arc_num()
    def maxflow(self, reuse_trees=False):
        """
        Perform the maxflow computation in the graph. Returns the capacity of
        the minimum cut or, equivalently, the maximum flow of the graph.

        If flag reuse_trees is true while calling maxflow(), then search trees
        are reused from previous maxflow computation, as described in

            "Efficiently Solving Dynamic Markov Random Fields Using Graph Cuts."
            Pushmeet Kohli and Philip H.S. Torr
            International Conference on Computer Vision (ICCV), 2005
        """
        return self.thisptr.maxflow(reuse_trees)
    def mark_node(self, i):
        """
        If flag reuse_trees is true while calling maxflow(), then search trees
        are reused from previous maxflow computation.

        In this case before calling maxflow() the user must
        specify which parts of the graph have changed by calling mark_node():
          add_tweights(i),set_trcap(i)    => call mark_node(i)
          add_edge(i,j),set_rcap(a)       => call mark_node(i); mark_node(j)

        This option makes sense only if a small part of the graph is changed.
        The initialization procedure goes only through marked nodes then.
        """
        self.thisptr.mark_node(i)
    def mark_grid_nodes(self, nodeids):
        """
        Mark nodes that have changed. This is equivalent to calling ``mark_node``
        for many nodes.
        """
        self.thisptr.mark_grid_nodes(nodeids)
    def get_segment(self, i):
        """Returns which segment the given node belongs to."""
        return self.thisptr.what_segment(i)
    def get_grid_segments(self, np.ndarray nodeids):
        """
        After the maxflow is computed, this function returns which
        segment the given nodes belong to. The output is a boolean array
        of the same shape than the input array ``nodeids``.

        This is equivalent to calling ``get_segment`` for many nodes, but much faster.
        """
        return self.thisptr.get_grid_segments(nodeids)
    def get_nx_graph(self):
        """
        Build a NetworkX DiGraph with the status of the maxflow network. The
        resulting graph will contain the terminal and non-terminal nodes and
        edges. The attribute ``weight`` of every edge will be set to its
        residual capacity. If the residual capacity of an edge is 0, the edge is
        not included in the DiGraph.

        The residual capacity for an edge is defined as the full capacity of the
        edge (set with the methods ``add_tedge``, ``add_edge`` and their
        corresponding grid equivalents, ``add_grid_tedges`` and
        ``add_grid_edges``) minus the amount of flow passing through it.
        Therefore, the weights of the DiGraph depend on the amount of flow
        passing through the network and, in turn, this depends on whether the
        call to ``get_nx_graph`` is done before or after calling
        ``GraphFloat.maxflow``.

        Before calling the ``GraphFloat.maxflow``, there is no flow and
        therefore the residual capacity of every edge is equal to its full
        capacity.

        After calling ``GraphFloat.maxflow``, a virtual flow traverses the
        network from the source node to the sink node, and the residual
        capacities will be lower than the full capacities. Note that in this
        case, since ``get_nx_graph`` ignores edges with residual capacity 0, the
        edges in the minimum cut will not be included in the final DiGraph.

        Note that this function is slow and should be used only for debugging
        purposes.

        This method requires the Python NetworkX package.
        """

        import networkx as nx
        g = nx.DiGraph()

        # Add non-terminal nodes
        g.add_nodes_from(range(self.get_node_count()))

        # Add non-terminal edges with capacities
        cdef int num_edges = self.get_edge_count()
        cdef uintptr_t e = self.thisptr.get_first_arc()

        cdef int n1
        cdef int n2
        cdef double w
        for i in range(num_edges):

            n1 = self.thisptr.get_arc_from(e)
            n2 = self.thisptr.get_arc_to(e)
            w = self.thisptr.get_rcap(e)

            if w != 0.0:
                if(g.has_edge(n1, n2)):
                    g[n1][n2]['weight'] += w
                else:
                    g.add_edge(n1, n2, weight=w)
            e = self.thisptr.get_next_arc(e)

        # Add terminal nodes
        g.add_nodes_from(['s', 't'])

        # Add terminal edges
        cdef int num_nodes = self.get_node_count()
        cdef double rcap
        cdef int segment
        for i in range(num_nodes):

            segment = self.thisptr.what_segment(i)

            g.nodes[i]['segment'] = segment

            rcap = self.thisptr.get_trcap(i)
            if rcap > 0.0:
                g.add_edge('s', i, weight=rcap)
            elif rcap < 0.0:
                g.add_edge(i, 't', weight=-rcap)

        return g


def moore_structure(ndim=2, directed=False):
    """
    Build a structure matrix corresponding to the Moore neighborhood with the
    given dimensionality ``ndim``.

    In an directed structure, only half of the neighbors are considered. In
    undirected structures, all the neighbors are considered. For example, in two
    dimensions, this is the matrix for an directed Moore structure::

        0 0 0
        0 0 1
        1 1 1

    The matrix for an undirected Moore structure is::

        1 1 1
        1 0 1
        1 1 1

    The directed structure is suitable for the add_grid_edges method of the
    Graph class when the ``symmetric`` parameter is True.
    """

    if not directed:
        return np.ones((3,)*ndim)

    flat = np.ones(3**ndim)
    flat[:3**ndim/2 + 1] = 0
    return np.reshape(flat, (3,)*ndim)

def vonNeumann_structure(ndim=2, directed=False):
    """
    Build a structure matrix corresponding to the von Neumann neighborhood with
    the given dimensionality ``ndim``.

    In an directed structure, only half of the neighbors are considered. In
    undirected structures, all the neighbors are considered. For example, in two
    dimensions, this is the matrix for an directed von Neumann structure::

        0 0 0
        0 0 1
        0 1 0

    The matrix for an undirected von Neumann structure is::

        0 1 0
        1 0 1
        0 1 0

    The directed structure is suitable for the add_grid_edges method of the
    Graph class when the ``symmetric`` parameter is True.
    """

    res = np.zeros((3,)*ndim)
    for i in range(ndim):

        idx = [1,]*ndim
        idx[i] = 2
        res[tuple(idx)] = 1

        if not directed:
            idx[i] = 0
            res[tuple(idx)] = 1

    return res
