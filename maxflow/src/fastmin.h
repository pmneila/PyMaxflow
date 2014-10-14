
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

#endif
