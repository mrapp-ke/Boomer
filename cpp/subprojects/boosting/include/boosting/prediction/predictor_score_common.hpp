/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/model/head_complete.hpp"
#include "common/model/head_partial.hpp"
#include "common/prediction/predictor_common.hpp"
#include "common/prediction/predictor_score.hpp"
#include "common/util/validation.hpp"

namespace boosting {

    static inline void applyHead(const CompleteHead& head, VectorView<float64>::iterator iterator) {
        CompleteHead::score_const_iterator scoreIterator = head.scores_cbegin();
        uint32 numElements = head.getNumElements();

        for (uint32 i = 0; i < numElements; i++) {
            iterator[i] += scoreIterator[i];
        }
    }

    static inline void applyHead(const PartialHead& head, VectorView<float64>::iterator iterator) {
        PartialHead::score_const_iterator scoreIterator = head.scores_cbegin();
        PartialHead::index_const_iterator indexIterator = head.indices_cbegin();
        uint32 numElements = head.getNumElements();

        for (uint32 i = 0; i < numElements; i++) {
            uint32 index = indexIterator[i];
            iterator[index] += scoreIterator[i];
        }
    }

    static inline void applyHead(const IHead& head, VectorView<float64>::iterator scoreIterator) {
        auto completeHeadVisitor = [=](const CompleteHead& head) {
            applyHead(head, scoreIterator);
        };
        auto partialHeadVisitor = [=](const PartialHead& head) {
            applyHead(head, scoreIterator);
        };
        head.visit(completeHeadVisitor, partialHeadVisitor);
    }

    static inline void applyRule(const RuleList::Rule& rule,
                                 VectorConstView<const float32>::const_iterator featureValuesBegin,
                                 VectorConstView<const float32>::const_iterator featureValuesEnd,
                                 VectorView<float64>::iterator scoreIterator) {
        const IBody& body = rule.getBody();

        if (body.covers(featureValuesBegin, featureValuesEnd)) {
            const IHead& head = rule.getHead();
            applyHead(head, scoreIterator);
        }
    }

    static inline void applyRules(RuleList::const_iterator rulesBegin, RuleList::const_iterator rulesEnd,
                                  VectorConstView<const float32>::const_iterator featureValuesBegin,
                                  VectorConstView<const float32>::const_iterator featureValuesEnd,
                                  VectorView<float64>::iterator scoreIterator) {
        for (; rulesBegin != rulesEnd; rulesBegin++) {
            const RuleList::Rule& rule = *rulesBegin;
            applyRule(rule, featureValuesBegin, featureValuesEnd, scoreIterator);
        }
    }

    static inline void applyRule(const RuleList::Rule& rule,
                                 CsrConstView<const float32>::index_const_iterator featureIndicesBegin,
                                 CsrConstView<const float32>::index_const_iterator featureIndicesEnd,
                                 CsrConstView<const float32>::value_const_iterator featureValuesBegin,
                                 CsrConstView<const float32>::value_const_iterator featureValuesEnd,
                                 VectorView<float64>::iterator scoreIterator, float32* tmpArray1, uint32* tmpArray2,
                                 uint32 n) {
        const IBody& body = rule.getBody();

        if (body.covers(featureIndicesBegin, featureIndicesEnd, featureValuesBegin, featureValuesEnd, tmpArray1,
                        tmpArray2, n)) {
            const IHead& head = rule.getHead();
            applyHead(head, scoreIterator);
        }
    }

    static inline void applyRules(RuleList::const_iterator rulesBegin, RuleList::const_iterator rulesEnd,
                                  uint32 numFeatures,
                                  CsrConstView<const float32>::index_const_iterator featureIndicesBegin,
                                  CsrConstView<const float32>::index_const_iterator featureIndicesEnd,
                                  CsrConstView<const float32>::value_const_iterator featureValuesBegin,
                                  CsrConstView<const float32>::value_const_iterator featureValuesEnd,
                                  VectorView<float64>::iterator scoreIterator) {
        float32* tmpArray1 = new float32[numFeatures];
        uint32* tmpArray2 = new uint32[numFeatures] {};
        uint32 n = 1;

        for (; rulesBegin != rulesEnd; rulesBegin++) {
            const RuleList::Rule& rule = *rulesBegin;
            applyRule(rule, featureIndicesBegin, featureIndicesEnd, featureValuesBegin, featureValuesEnd, scoreIterator,
                      &tmpArray1[0], &tmpArray2[0], n);
            n++;
        }

        delete[] tmpArray1;
        delete[] tmpArray2;
    }

    static inline void aggregatePredictedScores(const CContiguousConstView<const float32>& featureMatrix,
                                                RuleList::const_iterator rulesBegin, RuleList::const_iterator rulesEnd,
                                                CContiguousView<float64>& scoreMatrix, uint32 exampleIndex,
                                                uint32 predictionIndex) {
        applyRules(rulesBegin, rulesEnd, featureMatrix.values_cbegin(exampleIndex),
                   featureMatrix.values_cend(exampleIndex), scoreMatrix.values_begin(predictionIndex));
    }

    static inline void aggregatePredictedScores(const CsrConstView<const float32>& featureMatrix,
                                                RuleList::const_iterator rulesBegin, RuleList::const_iterator rulesEnd,
                                                CContiguousView<float64>& scoreMatrix, uint32 exampleIndex,
                                                uint32 predictionIndex) {
        uint32 numFeatures = featureMatrix.getNumCols();
        applyRules(rulesBegin, rulesEnd, numFeatures, featureMatrix.indices_cbegin(exampleIndex),
                   featureMatrix.indices_cend(exampleIndex), featureMatrix.values_cbegin(exampleIndex),
                   featureMatrix.values_cend(exampleIndex), scoreMatrix.values_begin(predictionIndex));
    }

    /**
     * An implementation of the type `PredictionDispatcher::IPredictionDelegate` that aggregates the scores that are
     * predicted by the individual rules in a model and stores them in a matrix.
     *
     * @tparam FeatureMatrix    The type of the feature matrix that provides row-wise access to the feature values of
     *                          the query examples
     * @tparam Model            The type of the rule-based model that is used to obtain predictions
     */
    template<typename FeatureMatrix, typename Model>
    class ScorePredictionDelegate final
        : public PredictionDispatcher<float64, FeatureMatrix, Model>::IPredictionDelegate {
        private:

            CContiguousView<float64>& scoreMatrix_;

        public:

            /**
             * @param scoreMatrix A reference to an object of type `CContiguousView` that should be used to store the
             *                    aggregated scores
             */
            ScorePredictionDelegate(CContiguousView<float64>& scoreMatrix) : scoreMatrix_(scoreMatrix) {}

            /**
             * @see `PredictionDispatcher::IPredictionDelegate::predictForExample`
             */
            void predictForExample(const FeatureMatrix& featureMatrix, typename Model::const_iterator rulesBegin,
                                   typename Model::const_iterator rulesEnd, uint32 threadIndex, uint32 exampleIndex,
                                   uint32 predictionIndex) const override {
                aggregatePredictedScores(featureMatrix, rulesBegin, rulesEnd, scoreMatrix_, exampleIndex,
                                         predictionIndex);
            }
    };

    /**
     * An implementation of the type `IScorePredictor` that allows to predict label-wise regression scores for given
     * query examples by summing up the scores that are predicted by individual rules in a rule-based model for each
     * label individually.
     *
     * @tparam FeatureMatrix    The type of the feature matrix that provides row-wise access to the feature values of
     *                          the query examples
     * @tparam Model            The type of the rule-based model that is used to obtain predictions
     */
    template<typename FeatureMatrix, typename Model>
    class ScorePredictor final : public IScorePredictor {
        private:

            class IncrementalPredictor final
                : public AbstractIncrementalPredictor<FeatureMatrix, Model, DensePredictionMatrix<float64>> {
                private:

                    DensePredictionMatrix<float64> predictionMatrix_;

                protected:

                    DensePredictionMatrix<float64>& applyNext(const FeatureMatrix& featureMatrix, uint32 numThreads,
                                                              typename Model::const_iterator rulesBegin,
                                                              typename Model::const_iterator rulesEnd) override {
                        ScorePredictionDelegate<FeatureMatrix, Model> delegate(predictionMatrix_);
                        PredictionDispatcher<float64, FeatureMatrix, Model>().predict(delegate, featureMatrix,
                                                                                      rulesBegin, rulesEnd, numThreads);
                        return predictionMatrix_;
                    }

                public:

                    IncrementalPredictor(const ScorePredictor& predictor, uint32 maxRules)
                        : AbstractIncrementalPredictor<FeatureMatrix, Model, DensePredictionMatrix<float64>>(
                          predictor.featureMatrix_, predictor.model_, predictor.numThreads_, maxRules),
                          predictionMatrix_(DensePredictionMatrix<float64>(predictor.featureMatrix_.getNumRows(),
                                                                           predictor.numLabels_, true)) {}
            };

            const FeatureMatrix& featureMatrix_;

            const Model& model_;

            const uint32 numLabels_;

            const uint32 numThreads_;

        public:

            /**
             * @param featureMatrix A reference to an object of template type `FeatureMatrix` that provides row-wise
             *                      access to the feature values of the query examples
             * @param model         A reference to an object of template type `Model` that should be used to obtain
             *                      predictions
             * @param numLabels     The number of labels to predict for
             * @param numThreads    The number of CPU threads to be used to make predictions for different query
             *                      examples in parallel. Must be at least 1
             */
            ScorePredictor(const FeatureMatrix& featureMatrix, const Model& model, uint32 numLabels, uint32 numThreads)
                : featureMatrix_(featureMatrix), model_(model), numLabels_(numLabels), numThreads_(numThreads) {}

            /**
             * @see `IPredictor::predict`
             */
            std::unique_ptr<DensePredictionMatrix<float64>> predict(uint32 maxRules) const override {
                uint32 numExamples = featureMatrix_.getNumRows();
                std::unique_ptr<DensePredictionMatrix<float64>> predictionMatrixPtr =
                  std::make_unique<DensePredictionMatrix<float64>>(numExamples, numLabels_, true);
                ScorePredictionDelegate<FeatureMatrix, Model> delegate(*predictionMatrixPtr);
                PredictionDispatcher<float64, FeatureMatrix, Model>().predict(
                  delegate, featureMatrix_, model_.used_cbegin(maxRules), model_.used_cend(maxRules), numThreads_);
                return predictionMatrixPtr;
            }

            /**
             * @see `IPredictor::canPredictIncrementally`
             */
            bool canPredictIncrementally() const override {
                return true;
            }

            /**
             * @see `IPredictor::createIncrementalPredictor`
             */
            std::unique_ptr<IIncrementalPredictor<DensePredictionMatrix<float64>>> createIncrementalPredictor(
              uint32 maxRules) const override {
                if (maxRules != 0) assertGreaterOrEqual<uint32>("maxRules", maxRules, 1);
                return std::make_unique<IncrementalPredictor>(*this, maxRules);
            }
    };

}
