"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides Cython wrappers for the type definitions of tuples, as well as corresponding utility functions.
"""
from boomer.common._arrays cimport uint32, float32, float64


cdef extern from "cpp/tuples.h" nogil:

    cdef struct IndexedFloat32:
        uint32 index
        float32 value

    cdef struct IndexedFloat32Array:
        IndexedFloat32* data
        uint32 numElements

    cdef struct IndexedFloat64:
        uint32 index
        float64 value


cdef extern from "cpp/tuples.h" namespace "tuples" nogil:

    int compareIndexedFloat32(const void* a, const void* b)

    int compareIndexedFloat64(const void* a, const void* b)
