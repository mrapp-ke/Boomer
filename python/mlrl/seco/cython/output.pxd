from mlrl.common.cython._types cimport uint8, uint32
from mlrl.common.cython.output cimport AbstractBinaryPredictor, IPredictor


cdef extern from "seco/output/predictor_classification_label_wise.hpp" namespace "seco" nogil:

    cdef cppclass LabelWiseClassificationPredictorImpl"seco::LabelWiseClassificationPredictor"(IPredictor[uint8]):

        # Constructors:

        LabelWiseClassificationPredictorImpl(uint32 numThreads) except +


cdef class LabelWiseClassificationPredictor(AbstractBinaryPredictor):

    # Attributes:

    cdef uint32 num_threads
