
#ifndef _GRID_H
#define _GRID_H

#include <vector>
#include <algorithm>
#include <functional>

// Some defines for backwards compatibility with previous APIs of NumPy
#ifndef NPY_ARRAY_FORCECAST
#define NPY_ARRAY_FORCECAST NPY_FORCECAST
#endif

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
        if((s & 1) == 0)
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

template<typename T>
std::vector<T> getVector(PyArrayObject* arr, int length)
{
    int arr_ndim = PyArray_NDIM(arr);
    npy_intp* arr_shape = PyArray_DIMS(arr);

    if(arr_ndim > 1)
        throw std::runtime_error("`periodic` array must have at most 1 dimension");

    if(arr_ndim == 0)
    {
        T value = *reinterpret_cast<T*>(PyArray_GetPtr(arr, NULL));
        return std::vector<T>(length, value);
    }

    // add_ndim == 1
    if(arr_shape[0] < length)
        throw std::runtime_error("the length of `periodic` must be equal to the number of dimensions of `nodeids`");

    std::vector<T> result(length);
    for(npy_intp i = 0; i < length; ++i)
        result[i] = *reinterpret_cast<T*>(PyArray_GetPtr(arr, &i));
    return result;
}

template <typename captype, typename tcaptype, typename flowtype>
void Graph<captype,tcaptype,flowtype>::add_grid_edges(PyArrayObject* _nodeids,
                        PyObject* _weights,
                        PyObject* _structure,
                        int symmetric,
                        PyObject* _periodic)
{
    typedef typename std::pair<std::vector<int>, captype> StructElem;
    typedef std::vector<StructElem> Structure;

    int ndim = PyArray_NDIM(_nodeids);
    PyArrayObject* nodeids = reinterpret_cast<PyArrayObject*>(PyArray_FROMANY((PyObject*)_nodeids, NPY_LONG, 0, 0, NPY_ITER_ALIGNED | NPY_ARRAY_FORCECAST));
    if(nodeids == NULL)
        throw std::runtime_error("The horror");

    PyArrayObject* weights = reinterpret_cast<PyArrayObject*>(PyArray_FROMANY(_weights, numpy_typemap<captype>::type, 0, ndim, NPY_ITER_ALIGNED | NPY_ARRAY_FORCECAST));
    if(weights == NULL)
    {
        Py_DECREF(nodeids);
        throw std::runtime_error("invalid number of dimensions for `weights`");
    }

    PyArrayObject* structureArr = reinterpret_cast<PyArrayObject*>(PyArray_FROMANY(_structure, numpy_typemap<captype>::type, 0, ndim, NPY_ITER_ALIGNED | NPY_ARRAY_FORCECAST));
    if(structureArr == NULL)
    {
        Py_DECREF(weights);
        Py_DECREF(nodeids);
        throw std::runtime_error("invalid number of dimensions for `structure`");
    }

    PyArrayObject* periodicArr = reinterpret_cast<PyArrayObject*>(PyArray_FROMANY(_periodic, numpy_typemap<bool>::type, 0, 1, NPY_ITER_ALIGNED | NPY_ARRAY_FORCECAST));
    if(periodicArr == NULL)
    {
        Py_DECREF(structureArr);
        Py_DECREF(weights);
        Py_DECREF(nodeids);
        throw std::runtime_error("invalid value for `periodic`");
    }

    npy_intp* shape = PyArray_DIMS(nodeids);

    // Extract the structure in a sparse format.
    Structure structure;
    std::vector<bool> periodic;
    try
    {
        getSparseStructure(structureArr, ndim, &structure);
        periodic = getVector<bool>(periodicArr, ndim);
    }
    catch(std::exception& e)
    {
            Py_DECREF(periodicArr);
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
    {
        Py_DECREF(structureArr);
        Py_DECREF(weights);
        Py_DECREF(nodeids);
        throw std::runtime_error("unknown error creating a NpyIter");
    }

    NpyIter_IterNextFunc* iternext = NpyIter_GetIterNext(iter, NULL);
    NpyIter_GetMultiIndexFunc* getMI = NpyIter_GetGetMultiIndex(iter, NULL);
    char** dataptr = NpyIter_GetDataPtrArray(iter);

    npy_intp* mi = new npy_intp[ndim];
    npy_intp* n_mi = new npy_intp[ndim];

    // Iterate over the full array.
    do
    {
        getMI(iter, mi);

        long i = *reinterpret_cast<long*>(dataptr[0]);
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
                    if(periodic[d])
                    {
                        n_mi[d] = n_mi[d] % shape[d];

                        // C++ has an awkward modulo operator.
                        // Compensate for negative values.
                        if(n_mi[d] < 0)
                            n_mi[d] = shape[d] + n_mi[d];
                    }
                    else
                        valid_neigh = false;
                }
            }
            if(!valid_neigh)
                continue;

            // Get the neighbor.
            long j = *reinterpret_cast<long*>(PyArray_GetPtr(nodeids, n_mi));

            captype capacity = w * it->second;

            add_edge(i, j, capacity, symmetric ? capacity : captype(0));
        }

    }while(iternext(iter));
    delete [] mi;
    delete [] n_mi;

    NpyIter_Deallocate(iter);

    Py_DECREF(periodicArr);
    Py_DECREF(structureArr);
    Py_DECREF(weights);
    Py_DECREF(nodeids);
}

template <typename captype, typename tcaptype, typename flowtype>
    inline int Graph<captype,tcaptype,flowtype>::get_arc_from(arc_id _a)
{
    arc* a = reinterpret_cast<arc*>(_a);
    assert(a >= arcs && a < arc_last);
    return (node_id) (a->sister->head - nodes);
}

template <typename captype, typename tcaptype, typename flowtype>
    inline int Graph<captype,tcaptype,flowtype>::get_arc_to(arc_id _a)
{
    arc* a = reinterpret_cast<arc*>(_a);
    assert(a >= arcs && a < arc_last);
    return (node_id) (a->head - nodes);
}

template <typename captype, typename tcaptype, typename flowtype>
void Graph<captype,tcaptype,flowtype>::mark_grid_nodes(PyArrayObject* nodeids)
{
    NpyIter* iter = NpyIter_New(nodeids, NPY_ITER_READONLY,  NPY_KEEPORDER, NPY_NO_CASTING, NULL);
    if (iter == NULL) {
        throw std::runtime_error("unknown error creating a NpyIter");
    }

    NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
    char** dataptr = NpyIter_GetDataPtrArray(iter);

    do {
        long node = *reinterpret_cast<long*>(dataptr[0]);
        mark_node(node);
    } while(iternext(iter));

    NpyIter_Deallocate(iter);
}

template <typename captype, typename tcaptype, typename flowtype>
void Graph<captype,tcaptype,flowtype>::add_grid_tedges(PyArrayObject* _nodeids,
                                                       PyObject* _sourcecaps,
                                                       PyObject* _sinkcaps)
{
    PyArrayObject* nodeids = reinterpret_cast<PyArrayObject*>(PyArray_FROMANY((PyObject*)_nodeids, NPY_LONG, 0, 0, NPY_ITER_ALIGNED | NPY_ARRAY_FORCECAST));
    int ndim = PyArray_NDIM(nodeids);

    PyArrayObject* sourcecaps = reinterpret_cast<PyArrayObject*>(PyArray_FROMANY(_sourcecaps, numpy_typemap<tcaptype>::type, 0, ndim, NPY_ITER_ALIGNED | NPY_ARRAY_FORCECAST));
    if(sourcecaps == NULL)
    {
        Py_DECREF(nodeids);
        throw std::runtime_error("invalid number of dimensions for sourcecaps");
    }
    PyArrayObject* sinkcaps = reinterpret_cast<PyArrayObject*>(PyArray_FROMANY(_sinkcaps, numpy_typemap<tcaptype>::type, 0, ndim, NPY_ITER_ALIGNED | NPY_ARRAY_FORCECAST));
    if(sinkcaps == NULL)
    {
        Py_DECREF(sourcecaps);
        Py_DECREF(nodeids);
        throw std::runtime_error("invalid number of dimensions for sinkcaps");
    }

    // Create the multiiterator.
    PyArrayObject* op[3] = {nodeids, sourcecaps, sinkcaps};
    npy_uint32 op_flags[3] = {NPY_ITER_READONLY, NPY_ITER_READONLY, NPY_ITER_READONLY};
    NpyIter* iter = NpyIter_MultiNew(3, op,
                                 0,
                                 NPY_KEEPORDER,
                                 NPY_NO_CASTING,
                                 op_flags,
                                 NULL);
    if(iter == NULL)
    {
        Py_DECREF(sinkcaps);
        Py_DECREF(sourcecaps);
        Py_DECREF(nodeids);
        throw std::runtime_error("unknown error creating a NpyIter");
    }

    NpyIter_IterNextFunc* iternext = NpyIter_GetIterNext(iter, NULL);
    char** dataptr = NpyIter_GetDataPtrArray(iter);
    do
    {
        long node = *reinterpret_cast<long*>(dataptr[0]);
        tcaptype src = *reinterpret_cast<tcaptype*>(dataptr[1]);
        tcaptype snk = *reinterpret_cast<tcaptype*>(dataptr[2]);
        add_tweights(node, src, snk);
    }while(iternext(iter));

    NpyIter_Deallocate(iter);
    Py_DECREF(sinkcaps);
    Py_DECREF(sourcecaps);
    Py_DECREF(nodeids);
}

template <typename captype, typename tcaptype, typename flowtype>
PyArrayObject* Graph<captype,tcaptype,flowtype>::get_grid_segments(PyArrayObject* nodeids)
{
    PyArrayObject* op[2] = {nodeids, NULL};
    npy_uint32 op_flags[2] = {NPY_ITER_READONLY, NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE};
    PyArray_Descr* op_dtypes[2] = {NULL, PyArray_DescrFromType(NPY_BOOL)};
    NpyIter* iter = NpyIter_MultiNew(2, op, 0, NPY_KEEPORDER, NPY_NO_CASTING, op_flags, op_dtypes);
    if(iter == NULL)
        throw std::runtime_error("unknown error creating a NpyIter");

    NpyIter_IterNextFunc* iternext = NpyIter_GetIterNext(iter, NULL);
    char** dataptr = NpyIter_GetDataPtrArray(iter);

    do
    {
        long node = *reinterpret_cast<long*>(dataptr[0]);
        *reinterpret_cast<bool*>(dataptr[1]) = what_segment(node) == SINK;
    }while(iternext(iter));

    PyArrayObject* res = NpyIter_GetOperandArray(iter)[1];
    Py_INCREF(res);
    NpyIter_Deallocate(iter);
    return res;
}

#endif
