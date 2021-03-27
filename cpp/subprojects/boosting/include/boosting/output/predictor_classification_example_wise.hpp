/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/output/predictor.hpp"
#include "common/input/label_vector.hpp"
#include "common/measures/measure_similarity.hpp"
#include <unordered_map>
#include <functional>


namespace boosting {

    /**
     * Allows to predict the labels of given query examples using an existing rule-based model that has been learned
     * using a boosting algorithm.
     *
     * For prediction, the scores that are provided by the individual rules, are summed up. For each query example, the
     * aggregated score vector is then compared to known label sets in order to obtain a distance measure. The label set
     * that is closest to the aggregated score vector is finally predicted.
     */
    class ExampleWiseClassificationPredictor : public IPredictor<uint8> {

        private:

            /**
             * Allows to compute hashes for objects of type `LabelVector`.
             */
            struct HashFunction {

                inline std::size_t operator()(const std::unique_ptr<LabelVector>& v) const {
                    std::size_t hash = (std::size_t) v->getNumElements();

                    for (auto it = v->indices_cbegin(); it != v->indices_cend(); it++) {
                        hash ^= *it + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                    }

                    return hash;
                }

            };

            /**
             * Allows to check whether two objects of type `LabelVector` are equal.
             */
            struct EqualsFunction {

                inline bool operator()(const std::unique_ptr<LabelVector>& lhs,
                                       const std::unique_ptr<LabelVector>& rhs) const {
                    if (lhs->getNumElements() != rhs->getNumElements()) {
                        return false;
                    }

                    auto it1 = lhs->indices_cbegin();

                    for (auto it2 = rhs->indices_cbegin(); it2 != rhs->indices_cend(); it2++) {
                        if (*it1 != *it2) {
                            return false;
                        }

                        it1++;
                    }

                    return true;
                }

            };

            typedef std::unordered_map<std::unique_ptr<LabelVector>, uint32, HashFunction, EqualsFunction> LabelVectorSet;

            LabelVectorSet labelVectors_;

            std::shared_ptr<ISimilarityMeasure> measurePtr_;

            uint32 numThreads_;

        public:

            /**
             * @param measurePtr    A shared pointer to an object of type `ISimilarityMeasure` that should be used to
             *                      quantify the similarity between predictions and known label vectors
             * @param numThreads    The number of CPU threads to be used to make predictions for different query
             *                      examples in parallel. Must be at least 1
             */
            ExampleWiseClassificationPredictor(std::shared_ptr<ISimilarityMeasure> measurePtr, uint32 numThreads);

            /**
             * A visitor function for handling objects of the type `LabelVector`.
             */
            typedef std::function<void(const LabelVector&)> LabelVectorVisitor;

            /**
             * Adds a known label vector that may be predicted for individual query examples.
             *
             * @param labelVectorPtr An unique pointer to an object of type `LabelVector`
             */
            void addLabelVector(std::unique_ptr<LabelVector> labelVectorPtr);

            /**
             * Invokes the given visitor function for each unique label vector that has been provided via the function `addLabelVector`.
             *
             * @param visitor The visitor function for handling objects of the type `LabelVector`
             */
            void visit(LabelVectorVisitor visitor) const;

            /**
             * Obtains predictions for different examples, based on predicted scores, and writes them to a given
             * prediction matrix.
             *
             * @param scoreMatrix       A reference to an object of type `CContiguousView` that stores the predicted
             *                          scores
             * @param predictionMatrix  A reference to an object of type `CContiguousView`, the predictions should be
             *                          written to. May contain arbitrary values
             */
            void transform(const CContiguousView<float64>& scoreMatrix, CContiguousView<uint8>& predictionMatrix) const;

            void predict(const CContiguousFeatureMatrix& featureMatrix, CContiguousView<uint8>& predictionMatrix,
                         const RuleModel& model) const override;

            void predict(const CsrFeatureMatrix& featureMatrix, CContiguousView<uint8>& predictionMatrix,
                         const RuleModel& model) const override;

    };

}
