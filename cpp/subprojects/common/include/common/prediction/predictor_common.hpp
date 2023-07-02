/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/prediction/predictor.hpp"
#include "omp.h"

/**
 * Allows to obtain predictions for multiple query examples by delegating the prediction for individual examples to
 * another class.
 *
 * @tparam T                The type of the predictions
 * @tparam FeatureMatrix    The type of the feature matrix that provides row-wise access to the feature values of the
 *                          query examples
 * @tparam Model            The type of the rule-based model that is used to obtain predictions
 */
template<typename T, typename FeatureMatrix, typename Model>
class PredictionDispatcher final {
    public:

        /**
         * Defines an interface for all classes, the prediction for individual examples can be delegated to by a
         * `PredictionDispatcher`.
         */
        class IPredictionDelegate {
            public:

                virtual ~IPredictionDelegate() {};

                /**
                 * Obtains predictions for a single query example.
                 *
                 * @param featureMatrix     A reference to an object of template type `FeatureMatrix` that provides
                 *                          row-wise access to the feature values of the query examples
                 * @param rulesBegin        An iterator of type `Model::const_iterator` to the first rule that should be
                 *                          used for prediction
                 * @param rulesEnd          An iterator of type `Model::const_iterator` to the last rule (exclusive)
                 *                          that should be used for prediction
                 * @param threadIndex       The index of the thread used for prediction
                 * @param exampleIndex      The index of the query example to predict for
                 * @param predictionIndex   The index of the row in the prediction matrix, where the predictions should
                 *                          be stored
                 */
                virtual void predictForExample(const FeatureMatrix& featureMatrix,
                                               typename Model::const_iterator rulesBegin,
                                               typename Model::const_iterator rulesEnd, uint32 threadIndex,
                                               uint32 exampleIndex, uint32 predictionIndex) const = 0;
        };

        /**
         * Obtains predictions for multiple query examples by delegating the prediction for individual examples to a
         * given `PredictionDelegate`.
         *
         * @param delegate      A reference to an object of type `IPredictionDelegate`, the prediction for individual
         *                      examples should be delegated to
         * @param featureMatrix A reference to an object of template type `FeatureMatrix` that provides row-wise access
         *                      to the feature values of the query examples
         * @param rulesBegin    An iterator of type `Model::const_iterator` to the first rule that should be used for
         *                      prediction
         * @param rulesEnd      An iterator of type `Model::const_iterator` to the last rule (exclusive) that should be
         *                      used for prediction
         * @param numThreads    The number of CPU threads to be used to make predictions for different query examples in
         *                      parallel. Must be at least 1
         */
        void predict(const IPredictionDelegate& delegate, const FeatureMatrix& featureMatrix,
                     typename Model::const_iterator rulesBegin, typename Model::const_iterator rulesEnd,
                     uint32 numThreads) const {
            uint32 numExamples = featureMatrix.getNumRows();
            const IPredictionDelegate* delegatePtr = &delegate;
            const FeatureMatrix* featureMatrixPtr = &featureMatrix;

#pragma omp parallel for firstprivate(numExamples) firstprivate(delegatePtr) firstprivate(rulesBegin) \
  firstprivate(rulesEnd) firstprivate(featureMatrixPtr) schedule(dynamic) num_threads(numThreads)
            for (int64 i = 0; i < numExamples; i++) {
                uint32 threadIndex = (uint32) omp_get_thread_num();
                delegatePtr->predictForExample(*featureMatrixPtr, rulesBegin, rulesEnd, threadIndex, i, i);
            }
        }
};

/**
 * Allows to obtain sparse binary predictions for multiple query examples by delegating the prediction for individual
 * examples to another class.
 *
 * @tparam FeatureMatrix    The type of the feature matrix that provides row-wise access to the feature values of the
 *                          query examples
 * @tparam Model            The type of the rule-based model that is used to obtain predictions
 */
template<typename FeatureMatrix, typename Model>
class BinarySparsePredictionDispatcher final {
    public:

        /**
         * Defines an interface for all classes, the prediction for individual examples can be delegated to by a
         * `BinarySparsePredictionDispatcher`.
         */
        class IPredictionDelegate {
            public:

                virtual ~IPredictionDelegate() {};

                /**
                 * Obtains predictions for a single query example.
                 *
                 * @param featureMatrix     A reference to an object of template type `FeatureMatrix` that provides
                 *                          row-wise access to the feature values of the query examples
                 * @param rulesBegin        An iterator of type `Model::const_iterator` to the first rule that should be
                 *                          used for prediction
                 * @param rulesEnd          An iterator of type `Model::const_iterator` to the last rule (exclusive)
                 *                          that should be used for prediction
                 * @param threadIndex       The index of the thread used for prediction
                 * @param exampleIndex      The index of the query example to predict for
                 * @param predictionIndex   The index of the row in the prediction matrix, where the predictions should
                 *                          be stored
                 * @return                  The number of non-zero predictions
                 */
                virtual uint32 predictForExample(const FeatureMatrix& featureMatrix,
                                                 typename Model::const_iterator rulesBegin,
                                                 typename Model::const_iterator rulesEnd, uint32 threadIndex,
                                                 uint32 exampleIndex, uint32 predictionIndex) const = 0;
        };

        /**
         * Obtains predictions for multiple query examples by delegating the prediction for individual examples to a
         * given `IPredictionDelegate`.
         *
         * @param delegate      A reference to an object of type `IPredictionDelegate`, the prediction for individual
         *                      examples should be delegated to
         * @param featureMatrix A reference to an object of template type `FeatureMatrix` that provides row-wise access
         *                      to the feature values of the query examples
         * @param rulesBegin    An iterator of type `Model::const_iterator` to the first rule that should be used for
         *                      prediction
         * @param rulesEnd      An iterator of type `Model::const_iterator` to the last rule (exclusive) that should be
         *                      used for prediction
         * @param numThreads    The number of CPU threads to be used to make predictions for different query examples in
         *                      parallel. Must be at least 1
         * @return              The total number of non-zero predictions
         */
        uint32 predict(const IPredictionDelegate& delegate, const FeatureMatrix& featureMatrix,
                       typename Model::const_iterator rulesBegin, typename Model::const_iterator rulesEnd,
                       uint32 numThreads) const {
            uint32 numExamples = featureMatrix.getNumRows();
            const IPredictionDelegate* delegatePtr = &delegate;
            const FeatureMatrix* featureMatrixPtr = &featureMatrix;
            uint32 numNonZeroElements = 0;

#pragma omp parallel for reduction(+ : numNonZeroElements) firstprivate(numExamples) firstprivate(delegatePtr) \
  firstprivate(rulesBegin) firstprivate(rulesEnd) firstprivate(featureMatrixPtr) schedule(dynamic) \
  num_threads(numThreads)
            for (int64 i = 0; i < numExamples; i++) {
                uint32 threadIndex = (uint32) omp_get_thread_num();
                numNonZeroElements +=
                  delegatePtr->predictForExample(*featureMatrixPtr, rulesBegin, rulesEnd, threadIndex, i, i);
            }

            return numNonZeroElements;
        }
};

/**
 * An abstract base class for all implementations of the class `IIncrementalPredictor`.
 *
 * @tparam FeatureMatrix    The type of the feature matrix that provides row-wise access to the feature values of the
 *                          query examples
 * @tparam Model            The type of the rule-based model that is used to obtain predictions
 * @tparam PredictionMatrix The type of the matrix that is used to store the predictions
 */
template<typename FeatureMatrix, typename Model, typename PredictionMatrix>
class AbstractIncrementalPredictor : public IIncrementalPredictor<PredictionMatrix> {
    private:

        const FeatureMatrix& featureMatrix_;

        const uint32 numThreads_;

        typename Model::const_iterator current_;

        typename Model::const_iterator end_;

    protected:

        /**
         * Must be implemented by subclasses in order to obtain predictions.
         *
         * @param featureMatrix A reference to an object of template type `FeatureMatrix` that provides row-wise access
         *                      to the feature values of the query examples
         * @param numThreads    The number of CPU threads to be used to make predictions for different query examples in
         *                      parallel. Must be at least 1
         * @param rulesBegin    An iterator of type `Model::const_iterator` to the first rule that should be used for
         *                      prediction
         * @param rulesEnd      An iterator of type `Model::const_iterator` to the last rule (exclusive) that should be
         *                      used for prediction
         * @return              A reference to an object of template type `PredictionMatrix` that stores the predictions
         */
        virtual PredictionMatrix& applyNext(const FeatureMatrix& featureMatrix, uint32 numThreads,
                                            typename Model::const_iterator rulesBegin,
                                            typename Model::const_iterator rulesEnd) = 0;

    public:

        /**
         * @param featureMatrix A reference to an object of template type `FeatureMatrix` that provides row-wise access
         *                      to the feature values of the query examples
         * @param model         A reference to an object of template type `Model` that should be used for prediction
         * @param numThreads    The number of CPU threads to be used to make predictions for different query examples in
         *                      parallel. Must be at least 1
         * @param maxRules      The maximum number of rules to be used for prediction. Must be at least 1 or 0, if the
         *                      number of rules should not be restricted
         */
        AbstractIncrementalPredictor(const FeatureMatrix& featureMatrix, const Model& model, uint32 numThreads,
                                     uint32 maxRules)
            : featureMatrix_(featureMatrix), numThreads_(numThreads), current_(model.used_cbegin(maxRules)),
              end_(model.used_cend(maxRules)) {}

        virtual ~AbstractIncrementalPredictor() override {};

        uint32 getNumNext() const override final {
            return (uint32) (end_ - current_);
        }

        PredictionMatrix& applyNext(uint32 stepSize) override final {
            typename Model::const_iterator next = current_ + std::min(stepSize, this->getNumNext());
            PredictionMatrix& predictionMatrix = this->applyNext(featureMatrix_, numThreads_, current_, next);
            current_ = next;
            return predictionMatrix;
        }
};
