/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/prediction/predictor_score_common.hpp"
#include "boosting/prediction/transformation_binary.hpp"
#include "common/data/arrays.hpp"
#include "common/data/matrix_c_contiguous.hpp"
#include "common/prediction/predictor_binary.hpp"

namespace boosting {

    /**
     * An implementation of the type `IBinaryPredictor` that allows to predict binary labels for given query examples by
     * summing up the scores that are predicted by individual rules in a rule-based model and transforming the
     * aggregated scores into binary predictions in {0, 1} according to an `IBinaryTransformation`.
     *
     * @tparam FeatureMatrix    The type of the feature matrix that provides row-wise access to the feature values of
     *                          the query examples
     * @tparam Model            The type of the rule-based model that is used to obtain predictions
     */
    template<typename FeatureMatrix, typename Model>
    class BinaryPredictor final : public IBinaryPredictor {
        private:

            class IncrementalPredictor final
                : public AbstractIncrementalPredictor<FeatureMatrix, Model, DensePredictionMatrix<uint8>> {
                private:

                    class IncrementalPredictionDelegate final
                        : public PredictionDispatcher<uint8, FeatureMatrix, Model>::IPredictionDelegate {
                        private:

                            CContiguousView<float64>& realMatrix_;

                            CContiguousView<uint8>& predictionMatrix_;

                            const IBinaryTransformation& binaryTransformation_;

                        public:

                            IncrementalPredictionDelegate(CContiguousView<float64>& realMatrix,
                                                          CContiguousView<uint8>& predictionMatrix,
                                                          const IBinaryTransformation& binaryTransformation)
                                : realMatrix_(realMatrix), predictionMatrix_(predictionMatrix),
                                  binaryTransformation_(binaryTransformation) {}

                            void predictForExample(const FeatureMatrix& featureMatrix,
                                                   typename Model::const_iterator rulesBegin,
                                                   typename Model::const_iterator rulesEnd, uint32 threadIndex,
                                                   uint32 exampleIndex, uint32 predictionIndex) const override {
                                ScorePredictionDelegate<FeatureMatrix, Model>(realMatrix_)
                                  .predictForExample(featureMatrix, rulesBegin, rulesEnd, threadIndex, exampleIndex,
                                                     predictionIndex);
                                binaryTransformation_.apply(realMatrix_.values_cbegin(predictionIndex),
                                                            realMatrix_.values_cend(predictionIndex),
                                                            predictionMatrix_.values_begin(predictionIndex),
                                                            predictionMatrix_.values_end(predictionIndex));
                            }
                    };

                    const std::shared_ptr<IBinaryTransformation> binaryTransformationPtr_;

                    DensePredictionMatrix<float64> realMatrix_;

                    DensePredictionMatrix<uint8> predictionMatrix_;

                protected:

                    DensePredictionMatrix<uint8>& applyNext(const FeatureMatrix& featureMatrix, uint32 numThreads,
                                                            typename Model::const_iterator rulesBegin,
                                                            typename Model::const_iterator rulesEnd) override {
                        if (binaryTransformationPtr_) {
                            IncrementalPredictionDelegate delegate(realMatrix_, predictionMatrix_,
                                                                   *binaryTransformationPtr_);
                            PredictionDispatcher<uint8, FeatureMatrix, Model>().predict(
                              delegate, featureMatrix, rulesBegin, rulesEnd, numThreads);
                        }

                        return predictionMatrix_;
                    }

                public:

                    IncrementalPredictor(const BinaryPredictor& predictor, uint32 maxRules,
                                         std::shared_ptr<IBinaryTransformation> binaryTransformationPtr)
                        : AbstractIncrementalPredictor<FeatureMatrix, Model, DensePredictionMatrix<uint8>>(
                          predictor.featureMatrix_, predictor.model_, predictor.numThreads_, maxRules),
                          binaryTransformationPtr_(binaryTransformationPtr),
                          realMatrix_(DensePredictionMatrix<float64>(predictor.featureMatrix_.getNumRows(),
                                                                     predictor.numLabels_,
                                                                     binaryTransformationPtr_ != nullptr)),
                          predictionMatrix_(DensePredictionMatrix<uint8>(predictor.featureMatrix_.getNumRows(),
                                                                         predictor.numLabels_,
                                                                         binaryTransformationPtr_ == nullptr)) {}
            };

            class PredictionDelegate final
                : public PredictionDispatcher<uint8, FeatureMatrix, Model>::IPredictionDelegate {
                private:

                    CContiguousView<float64>& realMatrix_;

                    CContiguousView<uint8>& predictionMatrix_;

                    const IBinaryTransformation& binaryTransformation_;

                public:

                    PredictionDelegate(CContiguousView<float64>& realMatrix, CContiguousView<uint8>& predictionMatrix,
                                       const IBinaryTransformation& binaryTransformation)
                        : realMatrix_(realMatrix), predictionMatrix_(predictionMatrix),
                          binaryTransformation_(binaryTransformation) {}

                    void predictForExample(const FeatureMatrix& featureMatrix,
                                           typename Model::const_iterator rulesBegin,
                                           typename Model::const_iterator rulesEnd, uint32 threadIndex,
                                           uint32 exampleIndex, uint32 predictionIndex) const override {
                        uint32 numLabels = realMatrix_.getNumCols();
                        CContiguousView<float64>::value_iterator realIterator = realMatrix_.values_begin(threadIndex);
                        setArrayToZeros(realIterator, numLabels);
                        ScorePredictionDelegate<FeatureMatrix, Model>(realMatrix_)
                          .predictForExample(featureMatrix, rulesBegin, rulesEnd, threadIndex, exampleIndex,
                                             threadIndex);
                        binaryTransformation_.apply(realIterator, realMatrix_.values_end(threadIndex),
                                                    predictionMatrix_.values_begin(predictionIndex),
                                                    predictionMatrix_.values_end(predictionIndex));
                    }
            };

            const FeatureMatrix& featureMatrix_;

            const Model& model_;

            const uint32 numLabels_;

            const uint32 numThreads_;

            const std::shared_ptr<IBinaryTransformation> binaryTransformationPtr_;

        public:

            /**
             * @param featureMatrix             A reference to an object of template type `FeatureMatrix` that provides
             *                                  row-wise access to the feature values of the query examples
             * @param model                     A reference to an object of template type `Model` that should be used to
             *                                  obtain predictions
             * @param numLabels                 The number of labels to predict for
             * @param numThreads                The number of CPU threads to be used to make predictions for different
             *                                  query examples in parallel. Must be at least 1
             * @param binaryTransformationPtr   An unique pointer to an object of type `IBinaryTransformation` that
             *                                  should be used to transform aggregated scores into binary predictions or
             *                                  a null pointer, if all labels should be predicted as irrelevant
             */
            BinaryPredictor(const FeatureMatrix& featureMatrix, const Model& model, uint32 numLabels, uint32 numThreads,
                            std::unique_ptr<IBinaryTransformation> binaryTransformationPtr)
                : featureMatrix_(featureMatrix), model_(model), numLabels_(numLabels), numThreads_(numThreads),
                  binaryTransformationPtr_(std::move(binaryTransformationPtr)) {}

            /**
             * @see `IPredictor::predict`
             */
            std::unique_ptr<DensePredictionMatrix<uint8>> predict(uint32 maxRules) const override {
                uint32 numExamples = featureMatrix_.getNumRows();
                std::unique_ptr<DensePredictionMatrix<uint8>> predictionMatrixPtr =
                  std::make_unique<DensePredictionMatrix<uint8>>(numExamples, numLabels_,
                                                                 binaryTransformationPtr_ == nullptr);

                if (binaryTransformationPtr_) {
                    CContiguousMatrix<float64> scoreMatrix(numThreads_, numLabels_);
                    PredictionDelegate delegate(scoreMatrix, *predictionMatrixPtr, *binaryTransformationPtr_);
                    PredictionDispatcher<uint8, FeatureMatrix, Model>().predict(
                      delegate, featureMatrix_, model_.used_cbegin(maxRules), model_.used_cend(maxRules), numThreads_);
                }

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
            std::unique_ptr<IIncrementalPredictor<DensePredictionMatrix<uint8>>> createIncrementalPredictor(
              uint32 maxRules) const override {
                if (maxRules != 0) assertGreaterOrEqual<uint32>("maxRules", maxRules, 1);
                return std::make_unique<IncrementalPredictor>(*this, maxRules, binaryTransformationPtr_);
            }
    };

    /**
     * An implementation of the type `ISparseBinaryPredictor` that allows to predict sparse binary labels for given
     * query examples by summing up the scores that are predicted by individual rules in a rule-based model and
     * transforming the aggregated scores into binary predictions in {0, 1} according to an `IBinaryTransformation`.
     *
     * @tparam FeatureMatrix    The type of the feature matrix that provides row-wise access to the feature values of
     *                          the query examples
     * @tparam Model            The type of the rule-based model that is used to obtain predictions
     */
    template<typename FeatureMatrix, typename Model>
    class SparseBinaryPredictor final : public ISparseBinaryPredictor {
        private:

            class IncrementalPredictor final
                : public AbstractIncrementalPredictor<FeatureMatrix, Model, BinarySparsePredictionMatrix> {
                private:

                    class IncrementalPredictionDelegate final
                        : public BinarySparsePredictionDispatcher<FeatureMatrix, Model>::IPredictionDelegate {
                        private:

                            CContiguousView<float64>& realMatrix_;

                            BinaryLilMatrix& predictionMatrix_;

                            const IBinaryTransformation& binaryTransformation_;

                        public:

                            IncrementalPredictionDelegate(CContiguousView<float64>& realMatrix,
                                                          BinaryLilMatrix& predictionMatrix,
                                                          const IBinaryTransformation& binaryTransformation)
                                : realMatrix_(realMatrix), predictionMatrix_(predictionMatrix),
                                  binaryTransformation_(binaryTransformation) {}

                            uint32 predictForExample(const FeatureMatrix& featureMatrix,
                                                     typename Model::const_iterator rulesBegin,
                                                     typename Model::const_iterator rulesEnd, uint32 threadIndex,
                                                     uint32 exampleIndex, uint32 predictionIndex) const override {
                                ScorePredictionDelegate<FeatureMatrix, Model>(realMatrix_)
                                  .predictForExample(featureMatrix, rulesBegin, rulesEnd, threadIndex, exampleIndex,
                                                     predictionIndex);
                                BinaryLilMatrix::row predictionRow = predictionMatrix_[predictionIndex];
                                predictionRow.clear();
                                binaryTransformation_.apply(realMatrix_.values_cbegin(predictionIndex),
                                                            realMatrix_.values_cend(predictionIndex), predictionRow);
                                return (uint32) predictionRow.size();
                            }
                    };

                    const std::shared_ptr<IBinaryTransformation> binaryTransformationPtr_;

                    DensePredictionMatrix<float64> realMatrix_;

                    BinaryLilMatrix predictionMatrix_;

                    std::unique_ptr<BinarySparsePredictionMatrix> predictionMatrixPtr_;

                protected:

                    BinarySparsePredictionMatrix& applyNext(const FeatureMatrix& featureMatrix, uint32 numThreads,
                                                            typename Model::const_iterator rulesBegin,
                                                            typename Model::const_iterator rulesEnd) override {
                        uint32 numNonZeroElements;

                        if (binaryTransformationPtr_) {
                            IncrementalPredictionDelegate delegate(realMatrix_, predictionMatrix_,
                                                                   *binaryTransformationPtr_);
                            numNonZeroElements = BinarySparsePredictionDispatcher<FeatureMatrix, Model>().predict(
                              delegate, featureMatrix, rulesBegin, rulesEnd, numThreads);
                        } else {
                            numNonZeroElements = 0;
                        }

                        predictionMatrixPtr_ = createBinarySparsePredictionMatrix(
                          predictionMatrix_, realMatrix_.getNumCols(), numNonZeroElements);
                        return *predictionMatrixPtr_;
                    }

                public:

                    IncrementalPredictor(const SparseBinaryPredictor& predictor, uint32 maxRules,
                                         std::shared_ptr<IBinaryTransformation> binaryTransformationPtr)
                        : AbstractIncrementalPredictor<FeatureMatrix, Model, BinarySparsePredictionMatrix>(
                          predictor.featureMatrix_, predictor.model_, predictor.numThreads_, maxRules),
                          binaryTransformationPtr_(binaryTransformationPtr),
                          realMatrix_(DensePredictionMatrix<float64>(predictor.featureMatrix_.getNumRows(),
                                                                     predictor.numLabels_,
                                                                     binaryTransformationPtr_ != nullptr)),
                          predictionMatrix_(BinaryLilMatrix(predictor.featureMatrix_.getNumRows())) {}
            };

            class PredictionDelegate final
                : public BinarySparsePredictionDispatcher<FeatureMatrix, Model>::IPredictionDelegate {
                private:

                    CContiguousView<float64>& realMatrix_;

                    BinaryLilMatrix& predictionMatrix_;

                    const IBinaryTransformation& binaryTransformation_;

                public:

                    PredictionDelegate(CContiguousView<float64>& realMatrix, BinaryLilMatrix& predictionMatrix,
                                       const IBinaryTransformation& binaryTransformation)
                        : realMatrix_(realMatrix), predictionMatrix_(predictionMatrix),
                          binaryTransformation_(binaryTransformation) {}

                    uint32 predictForExample(const FeatureMatrix& featureMatrix,
                                             typename Model::const_iterator rulesBegin,
                                             typename Model::const_iterator rulesEnd, uint32 threadIndex,
                                             uint32 exampleIndex, uint32 predictionIndex) const override {
                        uint32 numLabels = realMatrix_.getNumCols();
                        CContiguousView<float64>::value_iterator realIterator = realMatrix_.values_begin(threadIndex);
                        setArrayToZeros(realIterator, numLabels);
                        ScorePredictionDelegate<FeatureMatrix, Model>(realMatrix_)
                          .predictForExample(featureMatrix, rulesBegin, rulesEnd, threadIndex, exampleIndex,
                                             threadIndex);
                        BinaryLilMatrix::row predictionRow = predictionMatrix_[predictionIndex];
                        binaryTransformation_.apply(realIterator, realMatrix_.values_end(threadIndex), predictionRow);
                        return (uint32) predictionRow.size();
                    }
            };

            const FeatureMatrix& featureMatrix_;

            const Model& model_;

            const uint32 numLabels_;

            const uint32 numThreads_;

            const std::shared_ptr<IBinaryTransformation> binaryTransformationPtr_;

        public:

            /**
             * @param featureMatrix             A reference to an object of template type `FeatureMatrix` that provides
             *                                  row-wise access to the feature values of the query examples
             * @param model                     A reference to an object of template type `Model` that should be used to
             *                                  obtain predictions
             * @param numLabels                 The number of labels to predict for
             * @param numThreads                The number of CPU threads to be used to make predictions for different
             *                                  query examples in parallel. Must be at least 1
             * @param binaryTransformationPtr   An unique pointer to an object of type `IBinaryTransformation` that
             *                                  should be used to transform real-valued predictions into binary
             *                                  predictions or a null pointer, if no such object is available
             */
            SparseBinaryPredictor(const FeatureMatrix& featureMatrix, const Model& model, uint32 numLabels,
                                  uint32 numThreads, std::unique_ptr<IBinaryTransformation> binaryTransformationPtr)
                : featureMatrix_(featureMatrix), model_(model), numLabels_(numLabels), numThreads_(numThreads),
                  binaryTransformationPtr_(std::move(binaryTransformationPtr)) {}

            /**
             * @see `IPredictor::predict`
             */
            std::unique_ptr<BinarySparsePredictionMatrix> predict(uint32 maxRules) const override {
                uint32 numExamples = featureMatrix_.getNumRows();
                BinaryLilMatrix predictionMatrix(numExamples);
                uint32 numNonZeroElements;

                if (binaryTransformationPtr_) {
                    CContiguousMatrix<float64> scoreMatrix(numThreads_, numLabels_);
                    PredictionDelegate delegate(scoreMatrix, predictionMatrix, *binaryTransformationPtr_);
                    numNonZeroElements = BinarySparsePredictionDispatcher<FeatureMatrix, Model>().predict(
                      delegate, featureMatrix_, model_.used_cbegin(maxRules), model_.used_cend(maxRules), numThreads_);
                } else {
                    numNonZeroElements = 0;
                }

                return createBinarySparsePredictionMatrix(predictionMatrix, numLabels_, numNonZeroElements);
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
            std::unique_ptr<IIncrementalPredictor<BinarySparsePredictionMatrix>> createIncrementalPredictor(
              uint32 maxRules) const override {
                if (maxRules != 0) assertGreaterOrEqual<uint32>("maxRules", maxRules, 1);
                return std::make_unique<IncrementalPredictor>(*this, maxRules, binaryTransformationPtr_);
            }
    };

}
