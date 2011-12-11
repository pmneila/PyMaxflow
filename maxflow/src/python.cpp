
#include <iostream>
#include <vector>
#include <boost/python.hpp>

#define EXTMODULE_IMPORT_ARRAY
#include "pyarraymodule.h"

#include "core/graph.h"
#include "fastmin.h"
#include "grid.h"

namespace py = boost::python;

void incr_indices(npy_intp* ind, int ndim, const npy_intp* shape)
{
    // Update the index.
    for(int j = ndim-1; j >= 0; --j)
    {
        if(ind[j] + 1 < shape[j])
        {
            ++ind[j];
            break;
        }
        else
            ind[j] = 0;
    }
}

template<class GraphT>
py::object get_all_segments(GraphT& g)
{
    npy_intp num_nodes = g.get_node_num();
    PyArrayObject* res = reinterpret_cast<PyArrayObject*>(
                            PyArray_SimpleNew(1, &num_nodes, NPY_BOOL));
    
    for(int i = 0; i < num_nodes; ++i)
        *reinterpret_cast<bool*>(PyArray_GETPTR1(res, i)) = g.what_segment(i) == SINK;
    
    return py::object(res);
}

// A special implementation of maxflow to avoid resusing trees,
// a feature limited by the license.
template <class T>
inline T mymaxflow(Graph<T,T,T>& g)
{
    return g.maxflow();
}

template<class T>
void add_graph_class(py::dict cls_dict, py::object key, const std::string& suffix)
{
    typedef Graph<T,T,T> GraphT;
    
    void (*v1)(GraphT&, const PyArrayObject*, const T&, int) = &add_grid_edges_direction<GraphT>;
    void (*v2)(GraphT&, const PyArrayObject*, const T&, const T&, int) = &add_grid_edges_direction<GraphT>;
    void (*w1)(GraphT&, const PyArrayObject*, const PyArrayObject*, int) = &add_grid_edges_direction_local<GraphT>;
    void (*w2)(GraphT&, const PyArrayObject*, const PyArrayObject*, const PyArrayObject*, int) = &add_grid_edges_direction_local<GraphT>;
    
    py::object c = py::class_<GraphT>(("Graph"+suffix).c_str(), "Graph class for the min-cut/max-flow algorithm.",
            py::init<T, T>("Constructor. Create a graph.\n\n"
            "``est_node_num`` gives an estimate of the maximum number of non-terminal\n"
            "nodes that can be added to the graph, while ``est_edge_num`` is an\n"
            "estimate of the maximum number of non-terminal edges.\n"
            "\n"
            "**Important:** It is possible to add more nodes to the graph than\n"
            "est_node_num (and node_num_max can be zero). However, if the count\n"
            "is exceeded, then the internal memory is reallocated (increased\n"
            "by 50\%), which is expensive. Also, temporarily the amount of allocated\n"
            "memory would be more than twice than needed. Similarly for edges.\n",
            (py::arg("self"), py::arg("est_node_num")=0, py::arg("est_edge_num")=0)))
        .def("add_nodes", &GraphT::add_node,
            "Adds node(s) to the graph. By default, one node is added.\n"
            "If ``num``>1, then ``num`` nodes are inserted. The method returns\n"
            "the identifier of the first node added.\n\n"
            "**Important:** see note about the constructor.",
            (py::arg("self"), py::arg("num_nodes")=1))
        .def("add_edge", &GraphT::add_edge,
            "Adds a bidirectional edge between nodes ``i`` and ``j`` with the\n"
            "weights ``cap`` and ``rev_cap``.\n\n"
            "**Important:** see note about the constructor.",
            (py::arg("self"), py::arg("i"), py::arg("j"), py::arg("cap"), py::arg("rev_cap")))
        .def("add_tedge", &GraphT::add_tweights,
            "Add an edge 'SOURCE->i' with capacity ``cap_source`` and another edge\n"
            "'i->SINK' with capacity ``cap_sink``. This method can be called multiple\n"
            "times for each node. Capacities can be negative.\n\n"
            "**Note:** No internal memory is allocated by this call. The capacities\n"
            "of terminal edges are stored as a pair of values in each node.",
            (py::arg("self"), py::arg("node"), py::arg("cap_source"), py::arg("cap_sink")))
        .def("maxflow", &mymaxflow<T>,
            "Perform the maxflow computation in the graph. Returns the capacity of\n"
            "the minimum cut, which is equivalent to the maximum flow of the graph.",
            (py::arg("self")))
        .def("get_segment", &GraphT::what_segment,
            "Returns which segment the given node belongs to. This can be\n"
            "``maxflow.SOURCE`` or ``maxflow.SINK``.\n\n"
            "If a node can be assigned to both the source and the sink segments,\n"
            "then ``default_segment`` is returned.",
            (py::arg("self"), py::arg("node"), py::arg("default_segment")=SOURCE))
        .def("reset", &GraphT::reset, "Removes all nodes and edges.")
        .def("get_node_num", &GraphT::get_node_num, "Returns the number of non-terminal nodes.",
            (py::arg("self")))
        .def("get_edge_num", &GraphT::get_arc_num, "Returns the number of non-terminal edges.",
            (py::arg("self")))
        .def("get_all_segments", &get_all_segments<GraphT>,
            "After the maxflow is computed, this function returns which segment\n"
            "each node belongs to. The output is a boolean array with is True\n"
            "for those nodes which belong to the sink segment and False for those\n"
            "nodes which belong to the source segment.",
            (py::arg("self")))
        .def("get_grid_segments", &get_grid_segments<GraphT>,
            "After the maxflow is computed, this function returns which\n"
            "segment the given nodes belong to. The output is a boolean array\n"
            "of the same shape than the input array ``nodeids``.",
            (py::arg("self"), py::arg("nodeids")))
        .def("add_grid_tedges", &add_grid_tedges<GraphT>,
            "Add terminal edges to a grid of nodes, given their identifiers in\n"
            "``nodeids``. ``sourcecaps`` and ``sinkcaps`` are arrays with the\n"
            "capacities of the edges from the source node and to the sink node,\n"
            "respectively. The shape of all these arrays must be equal.",
            (py::arg("self"), py::arg("nodeids"), py::arg("sourcecaps"), py::arg("sinkcaps")))
        .def("add_grid_edges", &add_grid_edges<GraphT>,
            "Add edges in a grid of nodes of the same capacities for all the\n"
            "edges. The array ``capacity`` gives the capacity of all edges.\n"
            "Its shape must be equal than the shape of ``nodeids``.",
            (py::arg("self"), py::arg("nodeids"), py::arg("capacity")))
        .def("add_grid_edges_direction", v2,
            "",
            (py::arg("self"), py::arg("nodeids"), py::arg("capacity"), py::arg("rcapacity"), py::arg("direction")))
        .def("add_grid_edges_direction", v1,
            "",
            (py::arg("self"), py::arg("nodeids"), py::arg("capacity"), py::arg("direction")))
        .def("add_grid_edges_direction_local", w2,
            "Add edges in a grid of nodes. Each edge will have its own capacity\n"
            "and reverse capacity, and all edges will be created along the same\n"
            "direction. The array ``capacities`` must have the same shape than\n"
            "``nodeids``, except for the dimension ``direction``, where the\n"
            "size must be equal than the size of ``nodeids`` in that dimension - 1.\n\n"
            "The capacity given by ``capacities[i_1,...i_d,...,i_n]`` will be\n"
            "assigned to the edge between the nodes (i_1,...,i_d,...,i_n) and\n"
            "the (i_1,...,i_d+1,...,i_n), where i_d, is the index associated\n"
            "to the dimension ``direction``.",
            (py::arg("self"), py::arg("nodeids"), py::arg("capacities"), py::arg("rcapacities"), py::arg("direction")))
        .def("add_grid_edges_direction_local", w1,
            "This method, provided for convenience, behaves like the previous one.\n"
            "In this case the capacities and reverse capacities are equal.",
            (py::arg("self"), py::arg("nodeids"), py::arg("capacities"), py::arg("direction")));
    
    cls_dict.attr("__setitem__")(key, c);
}

void* extract_pyarray(PyObject* x)
{
    return PyObject_TypeCheck(x, &PyArray_Type) ? x : 0;
}

struct PyArrayObject_to_python
{
    static PyObject* convert(const PyArrayObject& obj)
    {
        return (PyObject*)&obj;
    }
};

BOOST_PYTHON_MODULE(_maxflow)
{
    import_array();
    
    // termtype enum.
    py::enum_<termtype>("termtype")
        .value("SOURCE", SOURCE)
        .value("SINK", SINK)
    ;
    
    py::dict cls_dict = py::dict();    
    // add_graph_class<int>(py::dict(), py::eval("int"), "Int");
    add_graph_class<long>(cls_dict, py::eval("int"), "Int");
    add_graph_class<double>(cls_dict, py::eval("float"), "Float");
    py::scope().attr("Graph") = cls_dict;
    
    py::def("abswap_grid_step", abswap,
        ".. note:: Unless you really need to, you should not call this function.\n\n"
        "Perform an iteration of the alpha-beta-swap algorithm.\n"
        "``labels`` is a N-dimensional array with shape S=(S_1,...,S_N)\n"
        "which holds the labels. The labels should be integer values between\n"
        "0 and L-1, where L is the number of labels. ``D`` should be an\n"
        "N+1-dimensional array with shape (L,S_1,...,S_N).\n"
        "D[l,p1,...,pn] is the unary energy of assigning the label l to the\n"
        "variable at the position [p1,...,pn].\n\n"
        "``V`` should be a two-dimensional array (a matrix) with shape (L,L).\n"
        "It encodes the binary term. V[l1,l2] is the energy of assigning the\n"
        "labels l1 and l2 to neighbor variables. Both ``D`` and ``V`` must be of\n"
        "the same type. ``alpha`` and ``beta`` are the variables that can be\n"
        "swapped in this step.\n\n"
        "This function modifies the ``labels`` array in-place and\n"
        "returns a tuple with the graph used for the step and\n"
        "the energy of the cut. Note that the energy of the cut is **NOT**\n"
        "the energy of the labeling, and cannot be used directly as the\n"
        "criterion of convergence.",
        (py::arg("alpha"), py::arg("beta"), py::arg("D"), py::arg("V"), py::arg("labels")));
    py::def("aexpansion_grid_step", aexpansion,
        ".. note:: Unless you really need to, you should not call this function.\n\n"
        "Perform an iteration of the alpha-expansion algorithm.\n"
        "``labels`` is a N-dimensional array with shape S=(S_1,...,S_N)\n"
        "which holds the labels. The labels should be integer values between\n"
        "0 and L-1, where L is the number of labels. ``D`` should be an\n"
        "N+1-dimensional array with shape (L,S_1,...,S_N).\n"
        "D[l,p1,...,pn] is the unary energy of assigning the label l to the\n"
        "variable at the position [p1,...,pn].\n\n"
        "``V`` should be a two-dimensional array (a matrix) with shape (L,L).\n"
        "It encodes the binary term. V[l1,l2] is the energy of assigning the\n"
        "labels l1 and l2 to neighbor variables. Both ``D`` and ``V`` must be of\n"
        "the same type. ``alpha`` indicates the variable that will be expanded\n"
        "in this step.\n\n"
        "This function modifies the ``labels`` array in-place and\n"
        "returns a tuple with the graph used for the step and\n"
        "the energy of the cut. Note that the energy of the cut **IS** the\n"
        "energy of the labeling, and can be used directly as the criterion\n"
        "of convergence.",
        (py::arg("alpha"), py::arg("D"), py::arg("V"), py::arg("labels")));
    
    // Automatic conversion from Python ndarray to C PyArrayObject.
    py::converter::registry::insert(&extract_pyarray, py::type_id<PyArrayObject>());
    py::to_python_converter<PyArrayObject, PyArrayObject_to_python>();
    
    // Add a documentation string for the package.
    py::scope().attr("__doc__") =
    "maxflow\n"
    "-------\n\n"
    "``maxflow`` is a Python module for max-flow/min-cut computations. It wraps\n"
    "the C++ maxflow library by Vladimir Kolmogorov, which implements the\n"
    "algorithm described in\n\n"
    "\tAn Experimental Comparison of Min-Cut/Max-Flow Algorithms for Energy\n"
    "\tMinimization in Vision. Yuri Boykov and Vladimir Kolmogorov. TPAMI.\n\n"
    "This module aims to simplifying the construction of graphs with complex\n"
    "layouts. It provides two Graph classes, ``Graph[int]`` and ``Graph[float]``,\n"
    "for integer and real data types.\n\n"
    "Example:\n\n"
    ">>> g = maxflow.Graph[int](2, 2)\n"
    ">>> g.add_nodes(2)\n"
    "0\n"
    ">>> g.add_edge(0, 1, 1, 2)\n"
    ">>> g.add_tedge(0, 2, 5)\n"
    ">>> g.add_tedge(1, 9, 4)\n"
    ">>> g.maxflow()\n"
    "8\n"
    ">>> g.get_segments()\n"
    "array([ True, False], dtype=bool)\n\n"
    "If you use this library for research purposes, you **MUST** cite the\n"
    "aforementioned paper in any resulting publication.";
}
