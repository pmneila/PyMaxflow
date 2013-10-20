
#include "pyarray_index.h"

#include <iterator>
#include <cstring>
#include <algorithm>

pyarray_index::pyarray_index(int ndim)
    : ndim(ndim)
{
    idx = new npy_intp[ndim];
    std::memset(idx, 0, ndim*sizeof(npy_intp));
}

pyarray_index::pyarray_index(int ndim, int value)
    : ndim(ndim)
{
    idx = new npy_intp[ndim];
    std::fill(idx, idx+ndim, value);    
}

pyarray_index::pyarray_index(int ndim, const npy_intp* idx)
    : ndim(ndim)
{
    this->idx = new npy_intp[ndim];
    std::memcpy(this->idx, idx, ndim*sizeof(npy_intp));
}

pyarray_index::pyarray_index(const pyarray_index& rhs)
    : ndim(rhs.ndim)
{
    idx = new npy_intp[ndim];
    std::memcpy(idx, rhs.idx, ndim*sizeof(npy_intp));
}

pyarray_index& pyarray_index::operator=(const pyarray_index& rhs)
{
    if(ndim != rhs.ndim)
    {
        ndim = rhs.ndim;
        delete [] idx;
        idx = new npy_intp[ndim];
    }
    std::memcpy(idx, rhs.idx, ndim*sizeof(npy_intp));
    
    return *this;
}

pyarray_index::~pyarray_index()
{
    delete [] idx;
}

bool pyarray_index::operator==(const pyarray_index& rhs) const
{
    if(ndim != rhs.ndim)
        return false;
    
    return std::equal(idx, idx+ndim, rhs.idx);
}

std::ostream& operator<<(std::ostream& os, const pyarray_index& idx)
{
    os << "(";
    std::copy(idx.idx, idx.idx+idx.ndim, std::ostream_iterator<int>(os, ", "));
    os << ")";
    return os;
}

pyarray_iterator::pyarray_iterator(const PyArrayObject* arr)
    : ind(PyArray_NDIM(arr), 0), lower(PyArray_NDIM(arr), 0),
    upper(PyArray_NDIM(arr), PyArray_DIMS(arr)), end(false)
{}

pyarray_iterator::pyarray_iterator(const pyarray_index& shape)
    : ind(shape.ndim, 0), lower(shape.ndim, 0), upper(shape), end(false)
{}

pyarray_iterator::pyarray_iterator(const pyarray_index& ind, const pyarray_index& shape)
    : ind(ind), lower(ind.ndim, 0), upper(shape), end(false)
{}

pyarray_iterator::pyarray_iterator(const pyarray_index& ind, 
                                    const pyarray_index& lower, const pyarray_index& upper)
    : ind(ind), lower(lower), upper(upper), end(false)
{}

pyarray_iterator::pyarray_iterator(const pyarray_iterator& rhs)
    : ind(rhs.ind), lower(rhs.lower), upper(rhs.upper), end(rhs.end)
{}

pyarray_iterator& pyarray_iterator::operator++()
{
    int j;
    for(j = ind.ndim-1; j >= 0; --j)
    {
        if(ind[j] + 1 < upper[j])
        {
            ++ind[j];
            break;
        }
        else
            ind[j] = lower[j];
    }
    end = j == -1;
    
    return *this;
}

pyarray_iterator pyarray_iterator::operator++(int)
{
    pyarray_iterator aux = *this;
    int j;
    for(j = ind.ndim-1; j >= 0; --j)
    {
        if(ind[j] + 1 < upper[j])
        {
            ++ind[j];
            break;
        }
        else
            ind[j] = lower[j];
    }
    end = j == -1;
    
    return *this;
}

moore_neighbor_index::moore_neighbor_index(const pyarray_iterator& iter)
    : center(iter.getIndex())
{
    const pyarray_index& iterupper = iter.getUpper();
    const pyarray_index& iterlower = iter.getLower();
    
    this->init_iter(iterlower, iterupper);
}

moore_neighbor_index::moore_neighbor_index(const PyArrayObject* arr, const pyarray_index& center)
    : center(center)
{
    pyarray_index shapelower(center.ndim, 0);
    pyarray_index shapeupper(center.ndim, PyArray_DIMS(arr));
    this->init_iter(shapelower, shapeupper);
}

moore_neighbor_index::moore_neighbor_index(const moore_neighbor_index& rhs)
    : center(rhs.center)
{
    iter = new pyarray_iterator(*rhs.iter);
}

moore_neighbor_index& moore_neighbor_index::operator=(const moore_neighbor_index& rhs)
{
    center = rhs.center;
    *iter = *rhs.iter;
    return *this;
}

moore_neighbor_index::~moore_neighbor_index()
{
    delete iter;
}

void moore_neighbor_index::init_iter(const pyarray_index& shapelower, const pyarray_index& shapeupper)
{
    int ndim = center.ndim;
    pyarray_index upper(ndim), lower(ndim);
    for(int i = 0; i < ndim; ++i)
    {
        lower[i] = center[i] - 1;
        if(center[i] == shapelower[i])
            ++lower[i];
        
        upper[i] = center[i] + 2;
        if(center[i] == shapeupper[i] - 1)
            --upper[i];
    }
    
    this->iter = new pyarray_iterator(lower, lower, upper);
}

moore_neighbor_index& moore_neighbor_index::operator++()
{
    ++(*iter);
    return *this;
}

moore_neighbor_index moore_neighbor_index::operator++(int)
{
    moore_neighbor_index aux = *this;
    ++(*iter);
    return aux;
}

pyarray_index moore_neighbor_index::getOffset() const
{
    pyarray_index offset(center.ndim);
    std::transform(iter->getIndex().begin(), iter->getIndex().end(),
                    center.begin(), offset.begin(), std::minus<npy_intp>());
    return offset;
}
