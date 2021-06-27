from libcpp.memory cimport shared_ptr


cdef extern from "common/head_refinement/head_refinement_factory.hpp" nogil:

    cdef cppclass IHeadRefinementFactory:
        pass


cdef extern from "common/head_refinement/head_refinement_single.hpp" nogil:

    cdef cppclass SingleLabelHeadRefinementFactoryImpl"SingleLabelHeadRefinementFactory"(IHeadRefinementFactory):
        pass


cdef extern from "common/head_refinement/head_refinement_full.hpp" nogil:

    cdef cppclass FullHeadRefinementFactoryImpl"FullHeadRefinementFactory"(IHeadRefinementFactory):
        pass


cdef class HeadRefinementFactory:

    # Attributes:

    cdef shared_ptr[IHeadRefinementFactory] head_refinement_factory_ptr


cdef class SingleLabelHeadRefinementFactory(HeadRefinementFactory):
    pass


cdef class FullHeadRefinementFactory(HeadRefinementFactory):
    pass
