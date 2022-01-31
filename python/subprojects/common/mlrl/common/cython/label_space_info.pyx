"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from libcpp.utility cimport move
from libcpp.memory cimport make_unique

SERIALIZATION_VERSION = 2


cdef class LabelSpaceInfo:
    """
    Provides information about the label space that may be used as a basis for making predictions.
    """

    cdef ILabelSpaceInfo* get_label_space_info_ptr(self):
        pass


cdef class NoLabelSpaceInfo(LabelSpaceInfo):
    """
    Does not provide any information about the label space.
    """

    cdef ILabelSpaceInfo* get_label_space_info_ptr(self):
        return self.label_space_info_ptr.get()

    def __reduce__(self):
        return (NoLabelSpaceInfo, (), ())

    def __setstate__(self, state):
        self.label_space_info_ptr = createNoLabelSpaceInfo()


cdef class LabelVectorSet(LabelSpaceInfo):
    """
    Stores a set of unique label vectors, as well as their frequency.
    """

    cdef ILabelSpaceInfo* get_label_space_info_ptr(self):
        return self.label_vector_set_ptr.get()

    cdef __serialize_label_vector(self, const LabelVector& label_vector):
        cdef list label_vector_state = []
        cdef uint32 num_elements = label_vector.getNumElements()
        cdef LabelVector.const_iterator iterator = label_vector.cbegin()
        cdef uint32 i, label_index

        for i in range(num_elements):
            label_index = iterator[i]
            label_vector_state.append(label_index)

        self.state.append(label_vector_state)

    cdef unique_ptr[LabelVector] __deserialize_label_vector(self, object label_vector_state):
        cdef uint32 num_elements = len(label_vector_state)
        cdef unique_ptr[LabelVector] label_vector_ptr = make_unique[LabelVector](num_elements)
        cdef LabelVector.iterator iterator = label_vector_ptr.get().begin()
        cdef uint32 i, label_index

        for i in range(num_elements):
            label_index = label_vector_state[i]
            iterator[i] = label_index

        return move(label_vector_ptr)

    def __reduce__(self):
        self.state = []
        self.label_vector_set_ptr.get().visit(
            wrapLabelVectorVisitor(<void*>self, <LabelVectorCythonVisitor>self.__serialize_label_vector))
        cdef object state = (SERIALIZATION_VERSION, self.state)
        self.state = None
        return (LabelVectorSet, (), state)

    def __setstate__(self, state):
        cdef int version = state[0]

        if version != SERIALIZATION_VERSION:
            raise AssertionError('Version of the serialized LabelVectorSet is ' + str(version) + ', expected '
                                 + str(SERIALIZATION_VERSION))

        cdef list label_vector_list = state[1]
        cdef uint32 num_label_vectors = len(label_vector_list)
        cdef unique_ptr[ILabelVectorSet] label_vector_set_ptr = createLabelVectorSet()
        cdef list label_vector_state
        cdef uint32 i

        for i in range(num_label_vectors):
            label_vector_state = label_vector_list[i]
            label_vector_set_ptr.get().addLabelVector(self.__deserialize_label_vector(label_vector_state))

        self.label_vector_set_ptr = move(label_vector_set_ptr)
