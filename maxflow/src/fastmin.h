
/**
 * Algorithms from "Fast approximate energy minimization via graph-cuts".
 */

#ifndef _FASTMIN_H
#define _FASTMIN_H

#include <Python.h>

#include "core/graph.h"

PyObject* abswap(int alpha, int beta, PyArrayObject* d,
                PyArrayObject* v, PyArrayObject* labels);
PyObject* aexpansion(int alpha, PyArrayObject* d,
                PyArrayObject* v, PyArrayObject* labels);

/// Type dispatcher.
#define DISPATCHER(name, parameters, call, typecheck) \
template<class T, class integer_types_it> \
struct name##_ \
{ \
    typedef Graph<T,T,T> GraphT; \
    typedef typename mpl::deref<integer_types_it>::type L; \
    typedef typename mpl::next<integer_types_it>::type next; \
    \
    inline static PyObject* apply parameters \
    { \
        if(PyArray_TYPE(typecheck) == mpl::at<numpy_typemap,L>::type::value) \
            return name<T,L>call; \
        else \
            return name##_<T,next>::apply call; \
    } \
}; \
template<class T> \
struct name##_<T, signed_integer_types_end> \
{ \
    inline static PyObject* apply parameters \
    { \
        throw std::runtime_error("invalid type for labels (should be any integer type)"); \
    } \
};

#endif
