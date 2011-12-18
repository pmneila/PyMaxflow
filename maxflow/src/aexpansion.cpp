/*
 * Alpha-expansion with a graph cut.
 *
 * Pablo Márquez Neila 2010
 */

#include <iostream>
#include <algorithm>
#include <numeric>
#include <functional>
#include <limits>
#include <boost/python.hpp>
#include "pyarraymodule.h"

#include "fastmin.h"
#include "core/graph.h"

namespace py = boost::python;

void incr_indices(npy_intp* ind, int ndim, const npy_intp* shape);

template<class T, class S>
py::object aexpansion(int alpha, PyArrayObject* d, PyArrayObject* v,
                        PyArrayObject* labels)
{
    typedef Graph<T,T,T> GraphT;
    
    // Size of the labels matrix.
    int ndim = PyArray_NDIM(labels);
    npy_intp* shape = PyArray_DIMS(labels);
    
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
    g->add_node(num_nodes);
    
    // Get the array v from v_f.
    // Esmooth<T> v(v_f);
    
    // For each pixel in labels...
    npy_intp* head_ind = new npy_intp[ndim+1];
    npy_intp* ind = &head_ind[1];
    npy_intp* nind = new npy_intp[ndim];
    std::fill(ind, ind+ndim, 0);
    for(int node_index = 0; node_index < num_nodes; ++node_index)
    {
        // Take the label of current pixel.
        S label = *reinterpret_cast<S*>(PyArray_GetPtr(labels, ind));
        // Discard pixels not in the set P_{ab}.
        head_ind[0] = alpha;
        T t1 = *reinterpret_cast<T*>(PyArray_GetPtr(d, head_ind));
        T t2 = std::numeric_limits<T>::max();
        if(label != alpha)
        {
            head_ind[0] = label;
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
                continue;
            }
            
            // If labels are different, add an extra node.
            T dist_label_nlabel = *reinterpret_cast<T*>(PyArray_GETPTR2(v, label, nlabel));
            T dist_nlabel_alpha = *reinterpret_cast<T*>(PyArray_GETPTR2(v, nlabel, alpha));
            int extra_index = g->add_node(1);
            g->add_tweights(extra_index, 0, dist_label_nlabel);
            g->add_edge(node_index, extra_index, dist_label_alpha, dist_label_alpha);
            g->add_edge(nnode_index, extra_index, dist_nlabel_alpha, dist_nlabel_alpha);
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
    return py::make_tuple(g, energy);
}

DISPATCHER(aexpansion, (int alpha, PyArrayObject* d, PyArrayObject* v, PyArrayObject* labels),
           (alpha, d, v, labels), labels)

// Access point for the aexpansion function.
py::object aexpansion(int alpha, PyArrayObject* d, PyArrayObject* v,
                PyArrayObject* labels)
{
    if(PyArray_TYPE(d) == PyArray_DOUBLE)
        return aexpansion_<double,signed_integer_types_begin>::apply(alpha, d, v, labels);
        //return aexpansion<double,unsigned char>(alpha, d, v, labels);
    else if(PyArray_TYPE(d) == PyArray_LONG)
        return aexpansion_<long,signed_integer_types_begin>::apply(alpha, d, v, labels);
        //return aexpansion<long,unsigned char>(alpha, d, v, labels);
    else
        throw std::runtime_error("the type of the unary term D is not valid (should be np.double or np.int)");
}
