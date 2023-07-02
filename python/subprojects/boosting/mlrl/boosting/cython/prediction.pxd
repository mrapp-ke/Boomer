from libcpp cimport bool


cdef extern from "boosting/prediction/predictor_probability_label_wise.hpp" namespace "boosting" nogil:

    cdef cppclass ILabelWiseProbabilityPredictorConfig:

        # Functions:

        bool isProbabilityCalibrationModelUsed() const

        ILabelWiseProbabilityPredictorConfig& setUseProbabilityCalibrationModel(bool useProbabilityCalibrationModel)


cdef extern from "boosting/prediction/predictor_probability_marginalized.hpp" namespace "boosting" nogil:

    cdef cppclass IMarginalizedProbabilityPredictorConfig:

        # Functions:

        bool isProbabilityCalibrationModelUsed() const

        IMarginalizedProbabilityPredictorConfig& setUseProbabilityCalibrationModel(bool useProbabilityCalibrationModel)


cdef extern from "boosting/prediction/predictor_binary_example_wise.hpp" namespace "boosting" nogil:

    cdef cppclass IExampleWiseBinaryPredictorConfig:

        # Functions:

        bool isBasedOnProbabilities() const

        IExampleWiseBinaryPredictorConfig& setBasedOnProbabilities(bool basedOnProbabilities)

        bool isProbabilityCalibrationModelUsed() const

        IExampleWiseBinaryPredictorConfig& setUseProbabilityCalibrationModel(bool useProbabilityCalibrationModel)


cdef extern from "boosting/prediction/predictor_binary_label_wise.hpp" namespace "boosting" nogil:

    cdef cppclass ILabelWiseBinaryPredictorConfig:

        # Functions:

        bool isBasedOnProbabilities() const

        ILabelWiseBinaryPredictorConfig& setBasedOnProbabilities(bool basedOnProbabilities)

        bool isProbabilityCalibrationModelUsed() const

        IExampleWiseBinaryPredictorConfig& setUseProbabilityCalibrationModel(bool useProbabilityCalibrationModel)


cdef extern from "boosting/prediction/predictor_binary_gfm.hpp" namespace "boosting" nogil:

    cdef cppclass IGfmBinaryPredictorConfig:

        # Functions:

        bool isProbabilityCalibrationModelUsed() const

        IGfmBinaryPredictorConfig& setUseProbabilityCalibrationModel(bool useProbabilityCalibrationModel)


cdef class LabelWiseProbabilityPredictorConfig:

    # Attributes:

    cdef ILabelWiseProbabilityPredictorConfig* config_ptr


cdef class MarginalizedProbabilityPredictorConfig:

    # Attributes:

    cdef IMarginalizedProbabilityPredictorConfig* config_ptr


cdef class ExampleWiseBinaryPredictorConfig:

    # Attributes:

    cdef IExampleWiseBinaryPredictorConfig* config_ptr


cdef class LabelWiseBinaryPredictorConfig:

    # Attributes:

    cdef ILabelWiseBinaryPredictorConfig* config_ptr


cdef class GfmBinaryPredictorConfig:

    # Attributes:

    cdef IGfmBinaryPredictorConfig* config_ptr
