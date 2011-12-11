
#ifndef _PYARRAY_INDEX_H
#define _PYARRAY_INDEX_H

#include <Python.h>
#include <stdexcept>

#include "pyarraymodule.h"

class pyarray_index
{
public:
    typedef npy_intp* iterator;
    typedef const npy_intp* const_iterator;
    
    npy_intp* idx;
    int ndim;
    
    explicit pyarray_index(int ndim);
    pyarray_index(int ndim, int value);
    pyarray_index(int ndim, const npy_intp* idx);
    pyarray_index(const pyarray_index& rhs);
    ~pyarray_index();
    
    pyarray_index& operator=(const pyarray_index& rhs);
    
    inline const npy_intp& operator[](int i) const {return idx[i];}
    inline npy_intp& operator[](int i) {return idx[i];}
    
    inline operator npy_intp*() {return &idx[0];}
    inline operator npy_intp*() const {return const_cast<npy_intp*>(&idx[0]);}
    
    inline iterator begin() {return idx;}
    inline iterator end() {return idx + ndim;}
    inline const_iterator begin() const {return idx;}
    inline const_iterator end() const {return idx + ndim;}
    
    bool operator==(const pyarray_index& rhs) const;
};

std::ostream& operator<<(std::ostream& os, const pyarray_index& idx);

/// Provisional iteration functionality until NpyIter is ready.
class pyarray_iterator
{
private:
    pyarray_index ind;
    pyarray_index lower;
    pyarray_index upper;
    bool end;
    
public:
    explicit pyarray_iterator(const PyArrayObject* arr);
    pyarray_iterator(const pyarray_index& shape);
    pyarray_iterator(const pyarray_index& ind, const pyarray_index& shape);
    pyarray_iterator(const pyarray_index& ind, const pyarray_index& lower, const pyarray_index& upper);
    pyarray_iterator(const pyarray_iterator& rhs);
    
    pyarray_iterator& operator++();
    pyarray_iterator operator++(int);
    
    inline const pyarray_index& getIndex() const {return ind;}
    
    pyarray_index getShape() const;
    
    inline int getNDim() const {return ind.ndim;}
    inline const pyarray_index& getUpper() const {return upper;}
    inline const pyarray_index& getLower() const {return lower;}
    
    inline bool atEnd() const {return end;}
};

/// Iterator through the Moore neighborhood of a given index.
class moore_neighbor_index
{
private:
    pyarray_index center;
    pyarray_iterator* iter;
    
    void init_iter(const pyarray_index& shapelower, const pyarray_index& shapeupper);
    
public:
    explicit moore_neighbor_index(const pyarray_iterator& idx);
    moore_neighbor_index(const PyArrayObject* arr, const pyarray_index& center);
    moore_neighbor_index(const moore_neighbor_index& rhs);
    ~moore_neighbor_index();
    
    moore_neighbor_index& operator=(const moore_neighbor_index& rhs);
    
    moore_neighbor_index& operator++();
    moore_neighbor_index operator++(int);
    
    inline const pyarray_index& getIndex() const {return iter->getIndex();}
    pyarray_index getOffset() const;
    
    inline int getNDim() const {return center.ndim;}    
    inline bool atEnd() const {return iter->atEnd();}
};

#endif
