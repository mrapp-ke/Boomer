"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from libcpp.memory cimport make_unique


cdef class Pruning:
    """
    A wrapper for the pure virtual C++ class `IPruning`.
    """
    pass


cdef class NoPruning(Pruning):
    """
    A wrapper for the C++ class `NoPruning`.
    """

    def __cinit__(self):
        self.pruning_ptr = <unique_ptr[IPruning]>make_unique[NoPruningImpl]()


cdef class IREP(Pruning):
    """
    A wrapper for the C++ class `IREP`.
    """

    def __cinit__(self):
        self.pruning_ptr = <unique_ptr[IPruning]>make_unique[IREPImpl]()
