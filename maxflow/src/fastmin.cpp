
#include <vector>
#include <map>
#include <algorithm>
#include <numeric>
#include <functional>
#include <limits>

#include "pyarraymodule.h"

#include "fastmin.h"
#include "core/graph.h"

// This avoids linker errors on Windows
#undef DL_IMPORT
#define DL_IMPORT(t) t
#include "_maxflow.h"

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

template <class T>
PyObject* build_graph_energy_tuple(Graph<T,T,T>* g, T energy);

template <>
PyObject* build_graph_energy_tuple<double>(Graph<double,double,double>* g, double energy)
{
    PyObject_GraphFloat* graph = PyObject_New(PyObject_GraphFloat, &GraphFloat);
    graph->thisptr = g;
    PyObject* res = Py_BuildValue("(d,O)", energy, graph);
    Py_XDECREF(graph);
    return res;
}

template <>
PyObject* build_graph_energy_tuple<long>(Graph<long,long,long>* g, long energy)
{
    PyObject_GraphInt* graph = PyObject_New(PyObject_GraphInt, &GraphInt);
    graph->thisptr = g;
    PyObject* res = Py_BuildValue("(l,O)", energy, graph);
    Py_XDECREF(graph);
    return res;
}

/*
 * Alpha-expansion with a graph cut.
 */
template<class T, class S>
PyObject* aexpansion(int alpha, PyArrayObject* d, PyArrayObject* v,
                        PyArrayObject* labels)
{
    typedef Graph<T,T,T> GraphT;

    // Size of the labels matrix.
    int ndim = PyArray_NDIM(labels);
    npy_intp* shape = PyArray_DIMS(labels);

    // Some shape checks.
    if(PyArray_NDIM(d) != ndim+1)
        throw std::runtime_error("the unary term matrix D must be SxL (S=shape of labels array, L=number of labels)");
    if(PyArray_NDIM(v) != 2 || PyArray_DIM(v, 0) != PyArray_DIM(v, 1))
        throw std::runtime_error("the binary term matrix V must be LxL (L=number of labels)");
    if(PyArray_DIM(v,0) != PyArray_DIM(d, ndim))
        throw std::runtime_error("the number of labels given by D differs from the number of labels given by V");
    if(PyArray_TYPE(v) != numpy_typemap<T>::type)
        throw std::runtime_error("the type for the binary term matrix V must match the type of the unary matrix D");
    if(!std::equal(shape, shape+ndim, PyArray_DIMS(d)))
        throw std::runtime_error("the shape of the labels array (S1,...,SN) must match the shape of the first dimensions of D (S1,...,SN,L)");

    // Create the graph.
    // The number of nodes and edges is unknown at this point,
    // so they are roughly estimated.
    int num_nodes = std::accumulate(shape, shape+ndim, 1, std::multiplies<int>());
    GraphT* g = new GraphT(num_nodes, 2*ndim*num_nodes);
    g->add_node(num_nodes);

    // Get the array v from v_f.
    // Esmooth<T> v(v_f);

    // For each pixel in labels...
    npy_intp* head_ind = new npy_intp[ndim+1];
    npy_intp* ind = head_ind;
    npy_intp* nind = new npy_intp[ndim];
    std::fill(ind, ind+ndim, 0);
    for(int node_index = 0; node_index < num_nodes; ++node_index)
    {
        // Take the label of current pixel.
        S label = *reinterpret_cast<S*>(PyArray_GetPtr(labels, ind));
        // Discard pixels not in the set P_{ab}.
        head_ind[ndim] = alpha;
        T t1 = *reinterpret_cast<T*>(PyArray_GetPtr(d, head_ind));
        T t2 = std::numeric_limits<T>::max();
        if(label != alpha)
        {
            head_ind[ndim] = label;
            t2 = *reinterpret_cast<T*>(PyArray_GetPtr(d, head_ind));
        }

        g->add_tweights(node_index, t1, t2);

        // Process the neighbors.
        for(int n = 0; n < ndim; ++n)
        {
            std::copy(ind, ind+ndim, nind);
            ++nind[n];
            // Discard bad neighbors.
            if(nind[n] >= shape[n])
                continue;

            // Neighbor index and label.
            int nnode_index = node_index + std::accumulate(shape+n+1, shape+ndim, 1, std::multiplies<int>());
            S nlabel = *reinterpret_cast<S*>(PyArray_GetPtr(labels, nind));

            T dist_label_alpha = *reinterpret_cast<T*>(PyArray_GETPTR2(v, label, alpha));
            if(label == nlabel)
            {
                g->add_edge(node_index, nnode_index, dist_label_alpha, dist_label_alpha);
            }
            else
            {
                // If labels are different, add an extra node.
                T dist_label_nlabel = *reinterpret_cast<T*>(PyArray_GETPTR2(v, label, nlabel));
                T dist_nlabel_alpha = *reinterpret_cast<T*>(PyArray_GETPTR2(v, nlabel, alpha));
                int extra_index = g->add_node(1);
                g->add_tweights(extra_index, 0, dist_label_nlabel);
                g->add_edge(node_index, extra_index, dist_label_alpha, dist_label_alpha);
                g->add_edge(nnode_index, extra_index, dist_nlabel_alpha, dist_nlabel_alpha);
            }
        }

        // Update the index.
        incr_indices(ind, ndim, shape);
    }

    // The graph cut.
    T energy = g->maxflow();

    // Update the labels with the maxflow result.
    std::fill(ind, ind+ndim, 0);
    for(int node_index = 0; node_index < num_nodes; ++node_index)
    {
        if(g->what_segment(node_index) == SINK)
            *reinterpret_cast<S*>(PyArray_GetPtr(labels, ind)) = alpha;

        // Update the index.
        incr_indices(ind, ndim, shape);
    }

    delete [] head_ind;
    delete [] nind;

    // Return the graph and the energy of the mincut.
    return build_graph_energy_tuple<T>(g, energy);
}

#define DISPATCH(name, ctype, parameters) \
case numpy_typemap<ctype>::type: \
    return name<T, ctype>parameters;

template<class T>
PyObject* aexpansion_(int alpha, PyArrayObject* d, PyArrayObject* v, PyArrayObject* labels)
{
    switch(PyArray_TYPE(labels))
    {
    DISPATCH(aexpansion, char, (alpha, d, v, labels));
    DISPATCH(aexpansion, short, (alpha, d, v, labels));
    DISPATCH(aexpansion, int, (alpha, d, v, labels));
    DISPATCH(aexpansion, long, (alpha, d, v, labels));
    DISPATCH(aexpansion, long long, (alpha, d, v, labels));
    default:
        throw std::runtime_error("invalid type for labels (should be any integer type)");
    }
}

// Access point for the aexpansion function.
PyObject* aexpansion(int alpha, PyArrayObject* d, PyArrayObject* v,
                PyArrayObject* labels)
{
    if(PyArray_TYPE(d) == NPY_DOUBLE)
        return aexpansion_<double>(alpha, d, v, labels);
    else if(PyArray_TYPE(d) == NPY_LONG)
        return aexpansion_<long>(alpha, d, v, labels);
    else
        throw std::runtime_error("the type of the unary term D is not valid (should be np.double or np.int)");
}

/*
 * Alpha-beta swap with a graph cut.
 */
template<class T, class S>
PyObject* abswap(int alpha, int beta, PyArrayObject* d, PyArrayObject* v,
                        PyArrayObject* labels)
{
    typedef Graph<T,T,T> GraphT;

    // Map graph node -> label index.
    std::vector<int> lookup;
    // Map label index -> graph node.
    std::map<int,int> reverse;

    // Size of the labels matrix.
    int ndim = PyArray_NDIM(labels);
    npy_intp* shape = PyArray_DIMS(labels);
    npy_intp* strides = PyArray_STRIDES(labels);

    // Some shape checks.
    if(PyArray_NDIM(d) != ndim+1)
        throw std::runtime_error("the unary term matrix D must be SxL (S=shape of labels array, L=number of labels)");
    if(PyArray_NDIM(v) != 2 || PyArray_DIM(v, 0) != PyArray_DIM(v, 1))
        throw std::runtime_error("the binary term matrix V must be LxL (L=number of labels)");
    if(PyArray_DIM(v,0) != PyArray_DIM(d,ndim))
        throw std::runtime_error("the number of labels given by D differs from the number of labels given by V");
    if(PyArray_TYPE(v) != numpy_typemap<T>::type)
        throw std::runtime_error("the type for the binary term matrix V must match the type of the unary matrix D");
    if(!std::equal(shape, shape+ndim, PyArray_DIMS(d)))
        throw std::runtime_error("the shape of the labels array (S1,...,SN) must match the shape of the first dimensions of D (S1,...,SN,L)");

    // Create the graph.
    // The number of nodes and edges is unknown at this point,
    // so they are roughly estimated.
    int num_nodes = std::accumulate(shape, shape+ndim, 1, std::multiplies<int>());
    GraphT* g = new GraphT(num_nodes, 2*ndim*num_nodes);

    // Get the distance V(a,b).
    T Vab = *reinterpret_cast<T*>(PyArray_GETPTR2(v, alpha, beta));

    // For each pixel in labels...
    npy_intp* head_ind = new npy_intp[ndim+1];
    npy_intp* ind = head_ind;
    npy_intp* nind = new npy_intp[ndim];
    std::fill(ind, ind+ndim, 0);

    for(int i = 0; i < num_nodes; ++i, incr_indices(ind, ndim, shape))
    {
        // Offset of the current pixel.
        //int labels_index = labels_indexbase + j * PyArray_STRIDE(labels, 1);
        int labels_index = std::inner_product(ind, ind+ndim, strides, 0);

        // Take the label of current pixel.
        S label = *reinterpret_cast<S*>(PyArray_BYTES(labels) + labels_index);

        // Discard pixels not in the set P_{ab}.
        if(label != alpha && label != beta)
            continue;

        int node_index = g->add_node(1);
        // Add to the lookup table.
        lookup.push_back(labels_index);
        // Add to the reverse map.
        reverse[labels_index] = node_index;

        // T-links weights initialization.
        head_ind[ndim] = alpha;
        T ta = *reinterpret_cast<T*>(PyArray_GetPtr(d, head_ind));
        head_ind[ndim] = beta;
        T tb = *reinterpret_cast<T*>(PyArray_GetPtr(d, head_ind));

        // Process the neighbors.
        for(int n = 0, dir = -1; n < ndim; n = dir == 1 ? n+1 : n, dir*=-1)
        {
            std::copy(ind, ind+ndim, nind);
            nind[n] += dir;

            // Discard bad neighbors.
            if(nind[n] < 0 || nind[n] >= shape[n])
                continue;

            int labels_neighbor = std::inner_product(nind, nind+ndim, strides, 0);
            S label2 = *reinterpret_cast<S*>(PyArray_BYTES(labels) + labels_neighbor);
            if(label2 != alpha && label2 != beta)
            {
                // Add the weights to the t-links for the neighbors
                // not in P_{ab}.
                ta += *reinterpret_cast<T*>(PyArray_GETPTR2(v, alpha, label2));
                tb += *reinterpret_cast<T*>(PyArray_GETPTR2(v, beta, label2));
            }
            else if(dir == -1)
                // Add edges to the neighbors in P_{ab}.
                g->add_edge(node_index, reverse[labels_neighbor], Vab, Vab);
        }

        g->add_tweights(node_index, ta, tb);
    }

    // The graph cut.
    T energy = g->maxflow();

    // Update the labels with the maxflow result.
    for(unsigned int i = 0; i < lookup.size(); ++i)
    {
        int labels_index = lookup[i];
        S* label = reinterpret_cast<S*>(PyArray_BYTES(labels) + labels_index);
        *label = g->what_segment(i) == SINK ? alpha : beta;
    }

    delete [] head_ind;
    delete [] nind;

    // Return the graph and the energy of the mincut.
    return build_graph_energy_tuple<T>(g, energy);
}

template<class T>
PyObject* abswap_(int alpha, int beta, PyArrayObject* d, PyArrayObject* v, PyArrayObject* labels)
{
    switch(PyArray_TYPE(labels))
    {
    DISPATCH(abswap, char, (alpha, beta, d, v, labels));
    DISPATCH(abswap, short, (alpha, beta, d, v, labels));
    DISPATCH(abswap, int, (alpha, beta, d, v, labels));
    DISPATCH(abswap, long, (alpha, beta, d, v, labels));
    DISPATCH(abswap, long long, (alpha, beta, d, v, labels));
    default:
        throw std::runtime_error("invalid type for labels (should be any integer type)");
    }
}

// Access point for the abswap function.
PyObject* abswap(int alpha, int beta, PyArrayObject* d, PyArrayObject* v,
                PyArrayObject* labels)
{
    if(PyArray_TYPE(d) == NPY_DOUBLE)
        return abswap_<double>(alpha, beta, d, v, labels);
    else if(PyArray_TYPE(d) == NPY_LONG)
        return abswap_<long>(alpha, beta, d, v, labels);
    else
        throw std::runtime_error("the type of the unary term D is not v (should be np.double or np.int)");
}
