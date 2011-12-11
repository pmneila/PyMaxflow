
#ifndef _GRID_H
#define _GRID_H

#include <boost/python.hpp>
#include <numeric>

#include "pyarray_index.h"

namespace py = boost::python;

/**

 */
template<class GraphT>
void add_grid_edges(GraphT& g, const PyArrayObject* nodeids,
            const typename GraphT::captype& cap)
{
    int ndim = PyArray_NDIM(nodeids);
    
    for(pyarray_iterator it(nodeids); !it.atEnd(); ++it)
    {
        npy_intp* coord = it.getIndex();
        int id1 = PyArray_SafeGet<int>(nodeids, coord);
        
        for(int d = 0; d < ndim; ++d)
        {
            if(coord[d] - 1 < 0)
                continue;
            
            --coord[d];
            int id2 = PyArray_SafeGet<int>(nodeids, coord);
            ++coord[d];
            
            g.add_edge(id1, id2, cap, cap);
        }
    }
}

/**
 * Set the capacities of the edges of a grid in a given direction.
 */
template<class GraphT>
void add_grid_edges_direction(GraphT& g, const PyArrayObject* nodeids, 
        const typename GraphT::captype& capacity,
        const typename GraphT::captype& rcapacity, int direction)
{
    typedef typename GraphT::captype captype;
        
    int ndim = PyArray_NDIM(nodeids);
    
    // Direction check.
    if(direction >= ndim)
        throw std::runtime_error("the given direction is greater than the number of dimensions");
    if(direction < 0)
        throw std::runtime_error("the direction cannot be negative");
    
    for(pyarray_iterator it(nodeids); !it.atEnd(); ++it)
    {
        npy_intp* coord = it.getIndex();
        if(coord[direction] - 1 < 0)
            continue;
        
        int id1 = PyArray_SafeGet<int>(nodeids, coord);
        --coord[direction];
        int id2 = PyArray_SafeGet<int>(nodeids, coord);
        ++coord[direction];
        
        g.add_edge(id2, id1, capacity, rcapacity);
    }
}

/**
 * Set the capacities of the edges of a grid in a given direction.
 */
template<class GraphT>
inline void add_grid_edges_direction(GraphT& g, const PyArrayObject* nodeids, 
         const typename GraphT::captype& capacity, int direction)
{
    add_grid_edges_direction<GraphT>(g, nodeids, capacity, capacity, direction);
}

/**
 * Set the capacities of the edges of a grid in a given direction.
 */
template<class GraphT>
void add_grid_edges_direction_local(GraphT& g, const PyArrayObject* nodeids, 
        const PyArrayObject* capacities, const PyArrayObject* rcapacities, int direction)
{
    typedef typename GraphT::captype captype;
        
    int ndim = PyArray_NDIM(nodeids);
    npy_intp* shape = PyArray_DIMS(nodeids);
    npy_intp* caps_shape = PyArray_DIMS(capacities);
    
    // Direction check.
    if(direction >= ndim)
        throw std::runtime_error("the given direction is greater than the number of dimensions");
    if(direction < 0)
        throw std::runtime_error("the direction cannot be negative");
    // Check the number of dimensions.
    if(ndim != PyArray_NDIM(capacities))
        throw std::runtime_error("invalid number of dimensions for the capacities array");
    // Shape checks.
    if(PyArray_NDIM(capacities) != PyArray_NDIM(rcapacities))
        throw std::runtime_error("capacities and rcapacities must have the same shape");
    if(!std::equal(caps_shape, caps_shape+ndim, PyArray_DIMS(rcapacities)))
        throw std::runtime_error("capacities and rcapacities must have the same shape");
    if(std::mismatch(shape, shape+direction, caps_shape, std::less_equal<int>()).first != shape+direction
            || std::mismatch(shape+direction+1, shape+ndim, caps_shape+direction+1, std::less_equal<int>()).first != shape+ndim)
        throw std::runtime_error("invalid shape for the capacities and rcapacities arrays");
    if(caps_shape[direction] < shape[direction]-1)
        throw std::runtime_error("invalid shape for the capacities and rcapacities arrays");
    
    for(pyarray_iterator it(nodeids); !it.atEnd(); ++it)
    {
        npy_intp* coord = it.getIndex();
        if(coord[direction] - 1 < 0)
            continue;
        
        int id1 = PyArray_SafeGet<int>(nodeids, coord);
        --coord[direction];
        int id2 = PyArray_SafeGet<int>(nodeids, coord);
        captype cap = PyArray_SafeGet<captype>(capacities, coord);
        captype rcap = PyArray_SafeGet<captype>(rcapacities, coord);
        ++coord[direction];
        
        g.add_edge(id2, id1, cap, rcap);
    }
}

/**
 * Set the capacities of the edges of a grid in a given direction.
 */
template<class GraphT>
inline void add_grid_edges_direction_local(GraphT& g, const PyArrayObject* nodeids, 
         const PyArrayObject* capacities, int direction)
{
    add_grid_edges_direction_local<GraphT>(g, nodeids, capacities, capacities, direction);
}

/**
 * Set the capacities of the terminal edges for multiple nodes.
 */
template<class GraphT>
void add_grid_tedges(GraphT& g, const PyArrayObject* nodeids,
            const PyArrayObject* sourcecaps, const PyArrayObject* sinkcaps)
{
    typedef typename GraphT::captype captype;
    
    int ndim = PyArray_NDIM(nodeids);
    npy_intp* shape = PyArray_DIMS(nodeids);
    
    // Shape checks.
    if(ndim != PyArray_NDIM(sourcecaps))
        throw std::runtime_error("the number of dimensions of the nodeids and capacities arrays must be equal");
    if(PyArray_NDIM(sourcecaps) != PyArray_NDIM(sinkcaps))
        throw std::runtime_error("the number of dimensions of source and sink arrays must be equal");
    if(!std::equal(shape, shape+ndim, PyArray_DIMS(sourcecaps)))
        throw std::runtime_error("nodeids and sourcecaps arrays must have the same shape");
    if(!std::equal(PyArray_DIMS(sourcecaps), PyArray_DIMS(sourcecaps)+ndim, PyArray_DIMS(sinkcaps)))
        throw std::runtime_error("source and sink capacity arrays must have the same shape");
    
    for(pyarray_iterator it(nodeids); !it.atEnd(); ++it)
    {
        npy_intp* coord = it.getIndex();
        int id = PyArray_SafeGet<int>(nodeids, coord);
        captype source = PyArray_SafeGet<captype>(sourcecaps, coord);
        captype sink = PyArray_SafeGet<captype>(sinkcaps, coord);
        g.add_tweights(id, source, sink);
    }
}

template<class GraphT>
py::object get_grid_segments(GraphT& g, const PyArrayObject* nodeids)
{
    int ndim = PyArray_NDIM(nodeids);
    npy_intp* shape = PyArray_DIMS(nodeids);
    PyArrayObject* res = reinterpret_cast<PyArrayObject*>(
                            PyArray_SimpleNew(ndim, shape, NPY_BOOL));
    
    for(pyarray_iterator it(nodeids); !it.atEnd(); ++it)
    {
        npy_intp* coord = it.getIndex();
        int id = PyArray_SafeGet<int>(nodeids, coord);
        *reinterpret_cast<bool*>(PyArray_GetPtr(res, coord)) = g.what_segment(id) == SINK;
    }
    
    return py::object(res);
}

#endif
