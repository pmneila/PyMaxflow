
# distutils: language = c++
# // distutils: sources = pyarray_index.cpp

import numpy as np
cimport numpy as np

cdef extern from "core/graph.h":
    cdef cppclass Graph[T,T,T]:
        Graph(int, int)
        int add_node(int)
        void add_edge(int, int, T, T) except +
        void add_tweights(int, T, T) except +
        void add_grid_edges(np.ndarray, int) except +
        void add_grid_tedges(np.ndarray, np.ndarray, np.ndarray)
        
        T maxflow()
        
        T what_segment(int)
        void get_grid_segments(np.ndarray)
    

cdef class GraphInt:
    cdef Graph[int,int,int]* thisptr
    def __cinit__(self, int est_node_num=0, int est_edge_num=0):
        self.thisptr = new Graph[int, int, int](est_node_num, est_edge_num)
    def __dealloc__(self):
        del self.thisptr
    def add_nodes(self, int num_nodes):
        return self.thisptr.add_node(num_nodes)
    def add_grid_nodes(self, shape):
        num_nodes = np.prod(shape)
        first = self.add_nodes(int(num_nodes))
        nodes = np.arange(first, first+num_nodes, dtype=np.int_)
        return np.reshape(nodes, shape)
    def add_edge(self, int i, int j, int capacity, int rcapacity):
        self.thisptr.add_edge(i, j, capacity, rcapacity)
    def add_tedge(self, int i, int cap_source, int cap_sink):
        self.thisptr.add_tweights(i, cap_source, cap_sink)
    def add_grid_edges(self, np.ndarray nodeids, int capacity):
        self.thisptr.add_grid_edges(nodeids, capacity)
    
    def maxflow(self):
        return self.thisptr.maxflow()
    
    def get_segment(self, i):
        return self.thisptr.what_segment(i)
    
    def get_grid_segments(self, np.ndarray nodeids):
        self.thisptr.get_grid_segments(nodeids)

cdef class GraphFloat:
    cdef Graph[float, float, float]* thisptr
    def __cinit__(self, int est_node_num=0, int est_edge_num=0):
        self.thisptr = new Graph[float, float, float](est_node_num, est_edge_num)
    def __dealloc__(self):
        del self.thisptr
    

cdef Graph_ = {}
