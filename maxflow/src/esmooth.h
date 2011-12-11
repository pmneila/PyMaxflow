
#ifndef _ESMOOTH_H
#define _ESMOOTH_H

#include <boost/python.hpp>
#include "pyarraymodule.h"

namespace py = boost::python;

/**
 * Esmooth holds information about the E_{smooth} term in the graph-cuts
 * based algorithms. The smoothing term represents the interactions between
 * labels, which may be different for each pair of variables.
 *
 * Esmooth is able to manage both globals distance functions V(f_p,f_q)
 * which are independent on the pair of variables, and local distance
 * functions V_{pq}(f_p,f_q) which depend on specific variables.
 */
template<class T>
class Esmooth
{
private:
    PyArrayObject* mV;
    bool mHasLocal;
    int mNumLabels;
    
    T* mGlobalDists;
    PyArrayObject** mLocalDists;
    
public:
    /**
     * Constructor.
     *
     * The numpy array v must be (N is the number of labels):
     *   - A symmetric NxN array of scalars. In that case,
     *     v[i,j] represents the global distance between labels
     *     i and j.
     *   - A symmetric NxN array of objects. Then, if v[i,j]
     *     is a scalar value, it represents the global distance
     *     between labels i and j. If v[i,j] is a numpy array,
     *     v[i,j][p_1,p_2] is the local distance between labels
     *     i and j of the variable (p_1,p_2) with its neighbors.
     */
    Esmooth(PyArrayObject* v)
        : mV(v), mGlobalDists(NULL), mLocalDists(NULL)
    {
        if(PyArray_NDIM(mV) < 2)
            throw std::runtime_error("the number of dimensions of the smoothing term should be 2");
        
        mNumLabels = PyArray_DIM(mV, 0);
        if(mNumLabels != PyArray_DIM(mV, 1))
            throw std::runtime_error("the smoothing term should be a squared matrix");
        
        mHasLocal = PyArray_TYPE(mV) == PyArray_OBJECT;
        if(!mHasLocal && PyArray_TYPE(mV) != mpl::at<numpy_typemap,T>::type::value)
            throw std::runtime_error("invalid type for V");
        
        if(mHasLocal)
        {
            mGlobalDists = new T[mNumLabels * mNumLabels];
            mLocalDists = new PyArrayObject*[mNumLabels * mNumLabels];
            
            for(int i = 0; i < mNumLabels; ++i)
                for(int j = 0; j < mNumLabels; ++j)
                {
                    int index = i * mNumLabels + j;
                    PyObject* obj = ::getitem<PyObject*>(mV, i, j);
                    py::extract<PyArrayObject*> extrArray(obj);
                    py::extract<T> extrValue(obj);
                    
                    mLocalDists[index] = 0;
                    if(extrArray.check())
                    {
                        PyArrayObject* arrobj = extrArray();
                        if(PyArray_TYPE(arrobj) != mpl::at<numpy_typemap,T>::type::value)
                            throw std::runtime_error("invalid type of a subarray found in the smoothing term");
                        
                        mLocalDists[index] = arrobj;
                    }
                    else if(extrValue.check())
                        mGlobalDists[index] = extrValue();
                    else
                        throw std::runtime_error("invalid type of a scalar found in the smoothing term");
                }
        }
    }
    
    ~Esmooth()
    {
        delete [] mGlobalDists;
        delete [] mLocalDists;
    }
    
    /**
     * Return the distance between labels l1 and l2 defined for
     * the variable (i,j) and its neighbor (ni,nj).
     */
    T getitem(int i, int j, int ni, int nj, int l1, int l2)
    {
        if(!mHasLocal)
            return ::getitem<T>(mV, l1, l2);
        
        int index = l1 * mNumLabels + l2;
        PyArrayObject* local = mLocalDists[index];
        if(!local)
            return mGlobalDists[index];
        
        return (::getitem<T>(local, i, j) + ::getitem<T>(local, ni, nj))/T(2);
    }
    
    /**
     * This function behaves as the previous one, but it is adapted for
     * 3D local functions.
     */
    T getitem(int i, int j, int k, int ni, int nj, int nk, int l1, int l2)
    {
        if(!mHasLocal)
            return ::getitem<T>(mV, l1, l2);
        
        int index = l1 * mNumLabels + l2;
        PyArrayObject* local = mLocalDists[index];
        if(!local)
            return mGlobalDists[index];
        
        return (::getitem<T>(local, i, j, k) + ::getitem<T>(local, ni, nj, 2))/T(2);
    }
    
    /**
     * Return the distance between the labels l1 and l2.
     * This is only valid when the defined distance is global,
     * i.e., when isLocal() returns true.
     */
    inline T& getitem(int l1, int l2)
    {
        return ::getitem<T>(mV, l1, l2);
    }
    
    /**
     * Return whether the defined distance is local, i.e.,
     * it depends on the variables and not only on the labels.
     */
    inline bool isLocal() const
    {
        return mHasLocal;
    }
};

#endif
