/*
 * Alpha-beta swap with a graph cut.
 *
 * Pablo Márquez Neila 2010
 */

#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <numeric>
#include <functional>
#include <boost/python.hpp>
#include "pyarraymodule.h"

#include "fastmin.h"
#include "core/graph.h"

namespace py = boost::python;

void incr_indices(npy_intp* ind, int ndim, const npy_intp* shape);

template<class T, class S>
py::object abswap(int alpha, int beta, PyArrayObject* d, PyArrayObject* v,
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
        throw std::runtime_error("the unary term matrix D must be LxS (L=number of labels, S=shape of labels array)");
    if(PyArray_NDIM(v) != 2 || PyArray_DIM(v, 0) != PyArray_DIM(v, 1))
        throw std::runtime_error("the binary term matrix V must be LxL (L=number of labels)");
    if(PyArray_DIM(v,0) != PyArray_DIM(d,0))
        throw std::runtime_error("the number of labels given by D differs from the number of labels given by V");
    if(PyArray_TYPE(v) != mpl::at<numpy_typemap,T>::type::value)
        throw std::runtime_error("the type for the binary term matrix V must match the type of the unary matrix D");
    if(!std::equal(shape, shape+ndim, &PyArray_DIMS(d)[1]))
        throw std::runtime_error("the shape of the labels array (S1,...,SN) must match the shape of the last dimensions of D (L,S1,...,SN)");
    
    // Create the graph.
    // The number of nodes and edges is unknown at this point,
    // so they are roughly estimated.
    int num_nodes = std::accumulate(shape, shape+ndim, 1, std::multiplies<int>());
    GraphT* g = new GraphT(num_nodes, 2*ndim*num_nodes);
    
    // Get the distance V(a,b).
    T Vab = *reinterpret_cast<T*>(PyArray_GETPTR2(v, alpha, beta));
    
    // For each pixel in labels...
    npy_intp* head_ind = new npy_intp[ndim+1];
    npy_intp* ind = &head_ind[1];
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
        head_ind[0] = alpha;
        T ta = *reinterpret_cast<T*>(PyArray_GetPtr(d, head_ind));
        head_ind[0] = beta;
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
    return py::make_tuple(g, energy);
}

DISPATCHER(abswap, (int alpha, int beta, PyArrayObject* d, PyArrayObject* v, PyArrayObject* labels),
           (alpha, beta, d, v, labels), labels);

// Access point for the abswap function.
py::object abswap(int alpha, int beta, PyArrayObject* d, PyArrayObject* v,
                PyArrayObject* labels)
{
    if(PyArray_TYPE(d) == PyArray_DOUBLE)
        return abswap_<double,signed_integer_types_begin>::apply(alpha, beta, d, v, labels);
        //return abswap<double,unsigned char>(alpha, beta, d, v, labels);
    else if(PyArray_TYPE(d) == PyArray_LONG)
        return abswap_<long,signed_integer_types_begin>::apply(alpha, beta, d, v, labels);
        //return abswap<long,unsigned char>(alpha, beta, d, v, labels);
    else
        throw std::runtime_error("the type of the unary term D is not v (should be np.double or np.int)");
}
