
# distutils: language = c++
# // distutils: sources = Rectangle.cpp

cdef extern from "core/graph.h":
    cdef cppclass Graph[T,T,T]:
        Graph(int, int)
        int add_node(int)

cdef extern from "grid.h":
    void add_grid_edges(Graph[T,T,T] self, c, T& cap)

cdef class GraphInt:
    cdef Graph[int,int,int]* thisptr
    def __cinit__(self, int est_node_num=0, int est_edge_num=0):
        self.thisptr = new Graph[int, int, int](est_node_num, est_edge_num)
    def __dealloc__(self):
        del self.thisptr
    def add_node(self, int num_nodes):
        return self.thisptr.add_node(num_nodes)
