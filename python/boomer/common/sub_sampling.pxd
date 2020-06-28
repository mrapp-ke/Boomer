from boomer.common._arrays cimport uint32, intp, float32
from boomer.common._random cimport RNG

from libcpp.pair cimport pair


cdef class InstanceSubSampling:

    # Functions:

    cdef pair[uint32[::1], uint32] sub_sample(self, intp num_examples, RNG rng)


cdef class Bagging(InstanceSubSampling):

    # Attributes:

    cdef readonly float32 sample_size

    # Functions:

    cdef pair[uint32[::1], uint32] sub_sample(self, intp num_examples, RNG rng)


cdef class RandomInstanceSubsetSelection(InstanceSubSampling):

    # Attributes
    cdef readonly float32 sample_size

    # Functions:

    cdef pair[uint32[::1], uint32] sub_sample(self, intp num_examples, RNG rng)


cdef class FeatureSubSampling:

    # Functions:

    cdef intp[::1] sub_sample(self, intp num_features, RNG rng)


cdef class RandomFeatureSubsetSelection(FeatureSubSampling):

    # Attributes:

    cdef readonly float32 sample_size

    # Functions:

    cdef intp[::1] sub_sample(self, intp num_features, RNG rng)


cdef class LabelSubSampling:

    # Functions:

    cdef intp[::1] sub_sample(self, intp num_labels, RNG rng)


cdef class RandomLabelSubsetSelection(LabelSubSampling):

    # Attributes:

    cdef readonly intp num_samples

    # Functions:

    cdef intp[::1] sub_sample(self, intp num_labels, RNG rng)
