from libcpp.cast cimport dynamic_cast
from libcpp.memory cimport unique_ptr

from mlrl.common.cython._types cimport uint32


cdef extern from "common/prediction/label_space_info.hpp" nogil:

    cdef cppclass ILabelSpaceInfo:
        pass


cdef extern from "common/prediction/label_space_info_no.hpp" nogil:

    cdef cppclass INoLabelSpaceInfo(ILabelSpaceInfo):
        pass


    unique_ptr[INoLabelSpaceInfo] createNoLabelSpaceInfo()


cdef extern from "common/input/label_vector.hpp" nogil:

    cdef cppclass LabelVector:

        ctypedef const uint32* const_iterator

        ctypedef uint32* iterator

        # Constructors:

        LabelVector(uint32 numElements)

        # Functions:

        uint32 getNumElements() const

        iterator begin()

        const_iterator cbegin() const


ctypedef void (*LabelVectorVisitor)(const LabelVector&, uint32)


cdef extern from "common/prediction/label_vector_set.hpp" nogil:

    cdef cppclass ILabelVectorSet(ILabelSpaceInfo):

        # Functions:

        void addLabelVector(unique_ptr[LabelVector] labelVectorPtr, uint32 frequency)

        void visit(LabelVectorVisitor) const


    unique_ptr[ILabelVectorSet] createLabelVectorSet()


ctypedef INoLabelSpaceInfo* NoLabelSpaceInfoPtr

ctypedef ILabelVectorSet* LabelVectorSetPtr


cdef extern from *:
    """
    #include "common/prediction/label_vector_set.hpp"


    typedef void (*LabelVectorCythonVisitor)(void*, const LabelVector&, uint32);

    static inline LabelVectorSet::LabelVectorVisitor wrapLabelVectorVisitor(
            void* self, LabelVectorCythonVisitor visitor) {
        return [=](const LabelVector& labelVector, uint32 frequency) {
            visitor(self, labelVector, frequency);
        };
    }
    """

    ctypedef void (*LabelVectorCythonVisitor)(void*, const LabelVector&, uint32 frequency)

    LabelVectorVisitor wrapLabelVectorVisitor(void* self, LabelVectorCythonVisitor visitor)


cdef class LabelSpaceInfo:

    # Functions:

    cdef ILabelSpaceInfo* get_label_space_info_ptr(self)


cdef class NoLabelSpaceInfo(LabelSpaceInfo):

    # Attributes:

    cdef unique_ptr[INoLabelSpaceInfo] label_space_info_ptr


cdef class LabelVectorSet(LabelSpaceInfo):

    # Attributes:

    cdef unique_ptr[ILabelVectorSet] label_vector_set_ptr

    cdef object state

    cdef object visitor

    # Functions:

    cdef __visit_label_vector(self, const LabelVector& label_vector, uint32 frequency)

    cdef __serialize_label_vector(self, const LabelVector& label_vector, uint32 frequency)

    cdef unique_ptr[LabelVector] __deserialize_label_vector(self, object label_vector_state)


cdef inline LabelSpaceInfo create_label_space_info(unique_ptr[ILabelSpaceInfo] label_space_info_ptr):
    cdef ILabelSpaceInfo* ptr = label_space_info_ptr.release()
    cdef ILabelVectorSet* label_vector_set_ptr = dynamic_cast[LabelVectorSetPtr](ptr)
    cdef INoLabelSpaceInfo* no_label_space_info_ptr
    cdef LabelVectorSet label_vector_set
    cdef NoLabelSpaceInfo no_label_space_info

    if label_vector_set_ptr != NULL:
        label_vector_set = LabelVectorSet.__new__(LabelVectorSet)
        label_vector_set.label_vector_set_ptr = unique_ptr[ILabelVectorSet](label_vector_set_ptr)
        return label_vector_set
    else:
        no_label_space_info_ptr = dynamic_cast[NoLabelSpaceInfoPtr](ptr)

        if no_label_space_info_ptr != NULL:
            no_label_space_info = NoLabelSpaceInfo.__new__(NoLabelSpaceInfo)
            no_label_space_info.label_space_info_ptr = unique_ptr[INoLabelSpaceInfo](no_label_space_info_ptr)
            return no_label_space_info
        else:
            del ptr
            raise RuntimeError('Encountered unsupported ILabelSpaceInfo object')
