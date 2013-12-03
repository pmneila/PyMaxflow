
#ifndef _GRID_H
#define _GRID_H

#include <boost/mpl/at.hpp>
#include <vector>

template<typename captype>
void getSparseStructure(PyArrayObject* structureArr,
    int ndim,
    std::vector<std::pair<std::vector<int>, captype> >* structure)
{
    typedef typename std::pair<std::vector<int>, captype> StructElem;
    
    // Check shape of the structure array.
    int structureNDIM = PyArray_NDIM(structureArr);
    npy_intp* structureShape = PyArray_DIMS(structureArr);
    int dimsdiff = ndim - structureNDIM;
    
    std::vector<int> center(ndim, 0);
    for(int i = 0; i < structureNDIM; ++i)
    {
        int s = structureShape[i];
        
        // Every dimension must be odd.
        if(s & 1 == 0)
        {
            throw std::runtime_error("the structure array must have an odd shape");
        }
        
        center[i + dimsdiff] = s >> 1;
    }
    
    // Create the sparse representation of the structure
    NpyIter* iter = NpyIter_New(structureArr,
                                NPY_ITER_READONLY | NPY_ITER_MULTI_INDEX,
                                NPY_KEEPORDER,
                                NPY_NO_CASTING,
                                NULL);
    
    if(iter == NULL)
        throw std::runtime_error("unknown error creating a NpyIter");
    
    NpyIter_IterNextFunc* iternext = NpyIter_GetIterNext(iter, NULL);
    NpyIter_GetMultiIndexFunc* getMI = NpyIter_GetGetMultiIndex(iter, NULL);
    char** dataptr = NpyIter_GetDataPtrArray(iter);
    
    npy_intp* mi = new npy_intp[ndim];
    std::fill(mi, mi+ndim, 0);
    do
    {
        captype v = *reinterpret_cast<captype*>(*dataptr);
        if(v == captype(0))
            continue;
        
        getMI(iter, &mi[dimsdiff]);
        if(std::equal(mi, mi+ndim, center.begin()))
            continue;
        
        // Subtract the center coord
        std::transform(mi, mi+ndim, center.begin(), mi, std::minus<int>());
        
        structure->push_back(StructElem(std::vector<int>(mi, mi+ndim), v));
    }while(iternext(iter));
    delete [] mi;
    
    NpyIter_Deallocate(iter);
}

template <typename captype, typename tcaptype, typename flowtype>
void Graph<captype,tcaptype,flowtype>::add_grid_edges(PyArrayObject* _nodeids,
                        PyObject* _weights,
                        PyObject* _structure,
                        int symmetric)
{
    typedef typename std::pair<std::vector<int>, captype> StructElem;
    typedef std::vector<StructElem> Structure;
    
    int ndim = PyArray_NDIM(_nodeids);
    PyArrayObject* nodeids = reinterpret_cast<PyArrayObject*>(PyArray_FROMANY((PyObject*)_nodeids, NPY_INT, 0, 0, NPY_ALIGNED));
    PyArrayObject* weights = reinterpret_cast<PyArrayObject*>(PyArray_FROMANY(_weights, (mpl::at<numpy_typemap,captype>::type::value), 0, 0, NPY_ALIGNED));
    PyArrayObject* structureArr = reinterpret_cast<PyArrayObject*>(PyArray_FROMANY(_structure, (mpl::at<numpy_typemap,captype>::type::value), 0, ndim, NPY_ALIGNED));
    
    npy_intp* shape = PyArray_DIMS(nodeids);
    
    if(structureArr == NULL)
        throw std::runtime_error("invalid number of dimensions");
    
    // Extract the structure in a sparse format.
    Structure structure;
    try
    {
        getSparseStructure(structureArr, ndim, &structure);
    }
    catch(std::exception& e)
    {
            Py_DECREF(structureArr);
            Py_DECREF(weights);
            Py_DECREF(nodeids);
            throw e;
    }
    
    // Create the edges
    PyArrayObject* op[2] = {nodeids, weights};
    npy_uint32 op_flags[2] = {NPY_ITER_READONLY, NPY_ITER_READONLY};
    NpyIter* iter = NpyIter_MultiNew(2, op,
                                     NPY_ITER_MULTI_INDEX,
                                     NPY_KEEPORDER,
                                     NPY_NO_CASTING,
                                     op_flags,
                                     NULL);
    
    if(iter == NULL)
        throw std::runtime_error("unknown error creating a NpyIter");
    
    NpyIter_IterNextFunc* iternext = NpyIter_GetIterNext(iter, NULL);
    NpyIter_GetMultiIndexFunc* getMI = NpyIter_GetGetMultiIndex(iter, NULL);
    char** dataptr = NpyIter_GetDataPtrArray(iter);
    
    npy_intp* mi = new npy_intp[ndim];
    npy_intp* n_mi = new npy_intp[ndim];
    
    // Iterate over the full array.
    do
    {
        getMI(iter, mi);
        
        int i = *reinterpret_cast<int*>(dataptr[0]);
        captype w = *reinterpret_cast<captype*>(dataptr[1]);
        
        // Neighbors...
        for(typename Structure::const_iterator it = structure.begin(); it != structure.end(); ++it)
        {
            const std::vector<int>& offset = it->first;
            std::transform(mi, mi+ndim, offset.begin(), n_mi, std::plus<int>());
            
            // Check if the neighbor is valid.
            bool valid_neigh = true;
            for(int d = 0; d < ndim && valid_neigh; ++d)
            {
                if(n_mi[d] < 0 || n_mi[d] >= shape[d])
                {
                    // TODO: Topology
                    valid_neigh = false;
                }
            }
            if(!valid_neigh)
                continue;
            
            // Get the neighbor.
            int j = *reinterpret_cast<int*>(PyArray_GetPtr(nodeids, n_mi));
            
            captype capacity = w * it->second;
            
            add_edge(i, j, capacity, symmetric ? capacity : captype(0));
        }
        
    }while(iternext(iter));
    delete [] mi;
    delete [] n_mi;
    
    NpyIter_Deallocate(iter);
    
    Py_DECREF(structureArr);
    Py_DECREF(weights);
    Py_DECREF(nodeids);
}

template <typename captype, typename tcaptype, typename flowtype> 
    inline int Graph<captype,tcaptype,flowtype>::get_arc_from(size_t _a)
{
    arc* a = reinterpret_cast<arc*>(_a);
    assert(a >= arcs && a < arc_last);
    return (node_id) (a->sister->head - nodes);
}

template <typename captype, typename tcaptype, typename flowtype> 
    inline int Graph<captype,tcaptype,flowtype>::get_arc_to(size_t _a)
{
    arc* a = reinterpret_cast<arc*>(_a);
    assert(a >= arcs && a < arc_last);
    return (node_id) (a->head - nodes);
}

/*template <typename captype, typename tcaptype, typename flowtype>
void Graph<captype,tcaptype,flowtype>::add_grid_edges(const PyArrayObject* nodeids,
            const captype& cap)
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
            
            add_edge(id1, id2, cap, cap);
        }
    }
}*/

template <typename captype, typename tcaptype, typename flowtype>
void Graph<captype,tcaptype,flowtype>::add_grid_tedges(const PyArrayObject* nodeids,
            const PyArrayObject* sourcecaps, const PyArrayObject* sinkcaps)
{
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
        add_tweights(id, source, sink);
    }
}

template <typename captype, typename tcaptype, typename flowtype>
PyArrayObject* Graph<captype,tcaptype,flowtype>::get_grid_segments(const PyArrayObject* nodeids)
{
    int ndim = PyArray_NDIM(nodeids);
    npy_intp* shape = PyArray_DIMS(nodeids);
    PyArrayObject* res = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(ndim, shape, NPY_BOOL));
    
    for(pyarray_iterator it(nodeids); !it.atEnd(); ++it)
    {
        npy_intp* coord = it.getIndex();
        int id = PyArray_SafeGet<int>(nodeids, coord);
        *reinterpret_cast<bool*>(PyArray_GetPtr(res, coord)) = what_segment(id) == SINK;
    }
    
    return res;
}

/**
 * Set the capacities of the edges of a grid in a given direction.
 */
template <typename captype, typename tcaptype, typename flowtype>
void Graph<captype,tcaptype,flowtype>::add_grid_edges_direction(const PyArrayObject* nodeids, 
        const captype& capacity,
        const captype& rcapacity,
        int direction)
{
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
        
        add_edge(id2, id1, capacity, rcapacity);
    }
}

// template <typename captype, typename tcaptype, typename flowtype>
// inline void Graph<captype,tcaptype,flowtype>::add_grid_edges_direction(const PyArrayObject* nodeids, 
//          const captype& capacity, int direction)
// {
//     add_grid_edges_direction(nodeids, capacity, capacity, direction);
// }

template <typename captype, typename tcaptype, typename flowtype>
void Graph<captype,tcaptype,flowtype>::add_grid_edges_direction_local(const PyArrayObject* nodeids, 
        const PyArrayObject* capacities, const PyArrayObject* rcapacities, int direction)
{
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
        
        add_edge(id2, id1, cap, rcap);
    }
}

#endif
