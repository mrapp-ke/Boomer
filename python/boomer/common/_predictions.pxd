"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

classes that store the predictions of rules, as well as corresponding quality scores.
"""
from boomer.common._arrays cimport uint32, float64


cdef extern from "cpp/predictions.h" nogil:

    cdef cppclass Prediction:

        # Attributes:

        uint32 numPredictions_

        uint32* labelIndices_

        float64* predictedScores_


    cdef cppclass PredictionCandidate(Prediction):

        # Constructors:

        PredictionCandidate(uint32 numPredictions, uint32* labelIndices, float64* predictedScores,
                            float64 overallQualityScore) except +

        # Attributes:

        float64 overallQualityScore_


    cdef cppclass LabelWisePredictionCandidate(PredictionCandidate):

        # Attributes:

        float64* qualityScores_
