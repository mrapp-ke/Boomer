#include "seco/statistics/statistics_label_wise_dense.hpp"
#include "common/data/arrays.hpp"
#include "common/statistics/statistics_subset_decomposable.hpp"
#include "seco/heuristics/confusion_matrices.hpp"
#include <cstdlib>


namespace seco {

    /**
     * Provides access to the elements of confusion matrices that are computed independently for each label using dense
     * data structures.
     */
    class LabelWiseStatistics final : public ILabelWiseStatistics {

        private:

            /**
             * Provides access to a subset of the confusion matrices that are stored by an instance of the class
             * `LabelWiseStatistics`.
             *
             * @tparam T The type of the vector that provides access to the indices of the labels that are included in
             *           the subset
             */
            template<class T>
            class StatisticsSubset final : public AbstractDecomposableStatisticsSubset {

                private:

                    const LabelWiseStatistics& statistics_;

                    std::unique_ptr<ILabelWiseRuleEvaluation> ruleEvaluationPtr_;

                    const T& labelIndices_;

                    float64* confusionMatricesCovered_;

                    float64* accumulatedConfusionMatricesCovered_;

                    float64* confusionMatricesSubset_;

                    float64* confusionMatricesCoverableSubset_;

                public:

                    /**
                     * @param statistics        A reference to an object of type `LabelWiseStatistics` that stores the
                     *                          confusion matrices
                     * @param ruleEvaluationPtr An unique pointer to an object of type `ILabelWiseRuleEvaluation` that
                     *                          should be used to calculate the predictions, as well as corresponding
                     *                          quality scores, of rules
                     * @param labelIndices      A reference to an object of template type `T` that provides access to
                     *                          the indices of the labels that are included in the subset
                     */
                    StatisticsSubset(const LabelWiseStatistics& statistics,
                                     std::unique_ptr<ILabelWiseRuleEvaluation> ruleEvaluationPtr, const T& labelIndices)
                        : statistics_(statistics), ruleEvaluationPtr_(std::move(ruleEvaluationPtr)),
                          labelIndices_(labelIndices) {
                        uint32 numPredictions = labelIndices.getNumElements();
                        confusionMatricesCovered_ =
                            (float64*) malloc(numPredictions * NUM_CONFUSION_MATRIX_ELEMENTS * sizeof(float64));
                        setArrayToZeros(confusionMatricesCovered_, numPredictions * NUM_CONFUSION_MATRIX_ELEMENTS);
                        accumulatedConfusionMatricesCovered_ = nullptr;
                        confusionMatricesSubset_ = statistics_.confusionMatricesSubset_;
                        confusionMatricesCoverableSubset_ = nullptr;
                    }

                    ~StatisticsSubset() {
                        free(confusionMatricesCovered_);
                        free(accumulatedConfusionMatricesCovered_);
                        free(confusionMatricesCoverableSubset_);
                    }

                    void addToMissing(uint32 statisticIndex, uint32 weight) override {
                        uint32 numLabels = statistics_.getNumLabels();

                        // Allocate arrays for storing the totals sums of gradients and Hessians, if necessary...
                        if (confusionMatricesCoverableSubset_ == nullptr) {
                            confusionMatricesCoverableSubset_ = (float64*) malloc(numLabels * sizeof(float64));
                            copyArray(confusionMatricesSubset_, confusionMatricesCoverableSubset_, numLabels);
                            confusionMatricesSubset_ = confusionMatricesCoverableSubset_;
                        }

                        // For each label, subtract the gradient and Hessian of the example at the given index (weighted
                        // by the given weight) from the total sum of gradients and Hessians...
                        uint32 offset = statisticIndex * numLabels;

                        for (uint32 c = 0; c < numLabels; c++) {
                            float64 labelWeight = statistics_.uncoveredLabels_[offset + c];

                            // Only uncovered labels must be considered...
                            if (labelWeight > 0) {
                                // Remove the current example and label from the confusion matrix that corresponds to
                                // the current label...
                                uint8 trueLabel = statistics_.labelMatrixPtr_->getValue(statisticIndex, c);
                                uint8 predictedLabel = statistics_.minorityLabels_[c];
                                uint32 element = getConfusionMatrixElement(trueLabel, predictedLabel);
                                confusionMatricesSubset_[c * NUM_CONFUSION_MATRIX_ELEMENTS + element] -= weight;
                            }
                        }
                    }

                    void addToSubset(uint32 statisticIndex, uint32 weight) override {
                        uint32 numLabels = statistics_.getNumLabels();
                        uint32 offset = statisticIndex * numLabels;
                        uint32 numPredictions = labelIndices_.getNumElements();
                        typename T::const_iterator indexIterator = labelIndices_.cbegin();

                        for (uint32 c = 0; c < numPredictions; c++) {
                            uint32 l = indexIterator[c];

                            // Only uncovered labels must be considered...
                            if (statistics_.uncoveredLabels_[offset + l] > 0) {
                                // Add the current example and label to the confusion matrix for the current label...
                                uint8 trueLabel = statistics_.labelMatrixPtr_->getValue(statisticIndex, l);
                                uint8 predictedLabel = statistics_.minorityLabels_[l];
                                uint32 element = getConfusionMatrixElement(trueLabel, predictedLabel);
                                confusionMatricesCovered_[c * NUM_CONFUSION_MATRIX_ELEMENTS + element] += weight;
                            }
                        }
                    }

                    void resetSubset() override {
                        uint32 numPredictions = labelIndices_.getNumElements();

                        // Allocate an array for storing the accumulated confusion matrices, if necessary...
                        if (accumulatedConfusionMatricesCovered_ == nullptr) {
                            accumulatedConfusionMatricesCovered_ =
                                (float64*) malloc(numPredictions * NUM_CONFUSION_MATRIX_ELEMENTS * sizeof(float64));
                            setArrayToZeros(accumulatedConfusionMatricesCovered_,
                                            numPredictions * NUM_CONFUSION_MATRIX_ELEMENTS);
                        }

                        // Reset the confusion matrix for each label to zero and add its elements to the accumulated
                        // confusion matrix...
                        for (uint32 c = 0; c < numPredictions; c++) {
                            uint32 offset = c * NUM_CONFUSION_MATRIX_ELEMENTS;
                            copyArray(&confusionMatricesCovered_[offset], &accumulatedConfusionMatricesCovered_[offset],
                                      NUM_CONFUSION_MATRIX_ELEMENTS);
                            setArrayToZeros(&confusionMatricesCovered_[offset], NUM_CONFUSION_MATRIX_ELEMENTS);
                        }
                    }

                    const ILabelWiseScoreVector& calculateLabelWisePrediction(bool uncovered,
                                                                              bool accumulated) override {
                        float64* confusionMatricesCovered =
                            accumulated ? accumulatedConfusionMatricesCovered_ : confusionMatricesCovered_;
                        return ruleEvaluationPtr_->calculateLabelWisePrediction(statistics_.minorityLabels_,
                                                                                statistics_.confusionMatricesTotal_,
                                                                                confusionMatricesSubset_,
                                                                                confusionMatricesCovered, uncovered);
                    }

            };

            uint32 numStatistics_;

            uint32 numLabels_;

            float64 sumUncoveredLabels_;

            std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr_;

            std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr_;

            float64* uncoveredLabels_;

            uint8* minorityLabels_;

            float64* confusionMatricesTotal_;

            float64* confusionMatricesSubset_;

        public:

            /**
             * @param ruleEvaluationFactoryPtr  A shared pointer to an object of type `ILabelWiseRuleEvaluationFactory`
             *                                  that allows to create instances of the class that is used for
             *                                  calculating the predictions, as well as corresponding quality scores, of
             *                                  rules
             * @param labelMatrixPtr            A shared pointer to an object of type `IRandomAccessLabelMatrix` that
             *                                  provides random access to the labels of the training examples
             * @param uncoveredLabels           A pointer to an array of type `float64`, shape
             *                                  `(numExamples, numLabels)`, indicating which examples and labels remain
             *                                  to be covered
             * @param sumUncoveredLabels        The sum of weights of all labels that remain to be covered
             * @param minorityLabels            A pointer to an array of type `uint8`, shape `(numLabels)`, indicating
             *                                  whether rules should predict individual labels as relevant (1) or
             *                                  irrelevant (0)
             */
            LabelWiseStatistics(std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr,
                                std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr, float64* uncoveredLabels,
                                float64 sumUncoveredLabels, uint8* minorityLabels)
                : numStatistics_(labelMatrixPtr->getNumRows()), numLabels_(labelMatrixPtr->getNumCols()),
                  sumUncoveredLabels_(sumUncoveredLabels), ruleEvaluationFactoryPtr_(ruleEvaluationFactoryPtr),
                  labelMatrixPtr_(labelMatrixPtr), uncoveredLabels_(uncoveredLabels), minorityLabels_(minorityLabels) {
                // The number of labels
                uint32 numLabels = this->getNumLabels();
                // A matrix that stores a confusion matrix, which takes into account all examples, for each label
                confusionMatricesTotal_ =
                    (float64*) malloc(numLabels * NUM_CONFUSION_MATRIX_ELEMENTS * sizeof(float64));
                // A matrix that stores a confusion matrix, which takes into account the examples covered by the
                // previous refinement of a rule, for each label
                confusionMatricesSubset_ =
                    (float64*) malloc(numLabels * NUM_CONFUSION_MATRIX_ELEMENTS * sizeof(float64));
            }

            ~LabelWiseStatistics() {
                free(uncoveredLabels_);
                free(minorityLabels_);
                free(confusionMatricesTotal_);
                free(confusionMatricesSubset_);
            }

            uint32 getNumStatistics() const override {
                return numStatistics_;
            }

            uint32 getNumLabels() const override {
                return numLabels_;
            }

            float64 getSumOfUncoveredLabels() const override {
                return sumUncoveredLabels_;
            }

            void setRuleEvaluationFactory(
                    std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr) override {
                ruleEvaluationFactoryPtr_ = ruleEvaluationFactoryPtr;
            }

            void resetSampledStatistics() override {
                uint32 numLabels = this->getNumLabels();
                uint32 numElements = numLabels * NUM_CONFUSION_MATRIX_ELEMENTS;
                setArrayToZeros(confusionMatricesTotal_, numElements);
                setArrayToZeros(confusionMatricesSubset_, numElements);
            }

            void addSampledStatistic(uint32 statisticIndex, uint32 weight) override {
                uint32 numLabels = this->getNumLabels();
                uint32 offset = statisticIndex * numLabels;

                for (uint32 c = 0; c < numLabels; c++) {
                    float64 labelWeight = uncoveredLabels_[offset + c];

                    // Only uncovered labels must be considered...
                    if (labelWeight > 0) {
                        // Add the current example and label to the confusion matrix that corresponds to the current
                        // label...
                        uint8 trueLabel = labelMatrixPtr_->getValue(statisticIndex, c);
                        uint8 predictedLabel = minorityLabels_[c];
                        uint32 element = getConfusionMatrixElement(trueLabel, predictedLabel);
                        uint32 i = c * NUM_CONFUSION_MATRIX_ELEMENTS + element;
                        confusionMatricesTotal_[i] += weight;
                        confusionMatricesSubset_[i] += weight;
                    }
                }
            }

            void resetCoveredStatistics() override {
                // Reset confusion matrices to 0...
                uint32 numLabels = this->getNumLabels();
                uint32 numElements = numLabels * NUM_CONFUSION_MATRIX_ELEMENTS;
                setArrayToZeros(confusionMatricesSubset_, numElements);
            }

            void updateCoveredStatistic(uint32 statisticIndex, uint32 weight, bool remove) override {
                uint32 numLabels = this->getNumLabels();
                uint32 offset = statisticIndex * numLabels;
                float64 signedWeight = remove ? -((float64) weight) : weight;

                for (uint32 c = 0; c < numLabels; c++) {
                    float64 labelWeight = uncoveredLabels_[offset + c];

                    // Only uncovered labels must be considered...
                    if (labelWeight > 0) {
                        // Add the current example and label to the confusion matrix that corresponds to the current
                        // label...
                        uint8 trueLabel = labelMatrixPtr_->getValue(statisticIndex, c);
                        uint8 predictedLabel = minorityLabels_[c];
                        uint32 element = getConfusionMatrixElement(trueLabel, predictedLabel);
                        confusionMatricesSubset_[c * NUM_CONFUSION_MATRIX_ELEMENTS + element] += signedWeight;
                    }
                }
            }

            std::unique_ptr<IStatisticsSubset> createSubset(const FullIndexVector& labelIndices) const override {
                std::unique_ptr<ILabelWiseRuleEvaluation> ruleEvaluationPtr =
                    ruleEvaluationFactoryPtr_->create(labelIndices);
                return std::make_unique<StatisticsSubset<FullIndexVector>>(*this, std::move(ruleEvaluationPtr),
                                                                           labelIndices);
            }

            std::unique_ptr<IStatisticsSubset> createSubset(const PartialIndexVector& labelIndices) const override {
                std::unique_ptr<ILabelWiseRuleEvaluation> ruleEvaluationPtr =
                    ruleEvaluationFactoryPtr_->create(labelIndices);
                return std::make_unique<StatisticsSubset<PartialIndexVector>>(*this, std::move(ruleEvaluationPtr),
                                                                              labelIndices);
            }

            void applyPrediction(uint32 statisticIndex, const FullPrediction& prediction) override {
                uint32 numLabels = this->getNumLabels();
                uint32 offset = statisticIndex * numLabels;
                uint32 numPredictions = prediction.getNumElements();
                FullPrediction::score_const_iterator scoreIterator = prediction.scores_cbegin();

                // Only the labels that are predicted by the new rule must be considered...
                for (uint32 c = 0; c < numPredictions; c++) {
                    uint8 predictedLabel = scoreIterator[c];
                    uint8 minorityLabel = minorityLabels_[c];

                    // Do only consider predictions that are different from the default rule's predictions...
                    if (predictedLabel == minorityLabel) {
                        uint32 i = offset + c;
                        float64 labelWeight = uncoveredLabels_[i];

                        if (labelWeight > 0) {
                            // Decrement the total sum of uncovered labels...
                            sumUncoveredLabels_ -= labelWeight;

                            // Mark the current example and label as covered...
                            uncoveredLabels_[i] = 0;
                        }
                    }
                }
            }

            void applyPrediction(uint32 statisticIndex, const PartialPrediction& prediction) override {
                uint32 numLabels = this->getNumLabels();
                uint32 offset = statisticIndex * numLabels;
                uint32 numPredictions = prediction.getNumElements();
                PartialPrediction::score_const_iterator scoreIterator = prediction.scores_cbegin();
                PartialPrediction::index_const_iterator indexIterator = prediction.indices_cbegin();

                // Only the labels that are predicted by the new rule must be considered...
                for (uint32 c = 0; c < numPredictions; c++) {
                    uint32 l = indexIterator[c];
                    uint8 predictedLabel = scoreIterator[c];
                    uint8 minorityLabel = minorityLabels_[l];

                    // Do only consider predictions that are different from the default rule's predictions...
                    if (predictedLabel == minorityLabel) {
                        uint32 i = offset + l;
                        float64 labelWeight = uncoveredLabels_[i];

                        if (labelWeight > 0) {
                            // Decrement the total sum of uncovered labels...
                            sumUncoveredLabels_ -= labelWeight;

                            // Mark the current example and label as covered...
                            uncoveredLabels_[i] = 0;
                        }
                    }
                }
            }

            float64 evaluatePrediction(uint32 statisticIndex, const IEvaluationMeasure& measure) const override {
                // TODO Support evaluation of predictions
                return 0;
            }

            std::unique_ptr<IHistogram> createHistogram(uint32 numBins) const override {
                //TODO Support creation of histograms
                return nullptr;
            }

    };

    DenseLabelWiseStatisticsFactory::DenseLabelWiseStatisticsFactory(
            std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr,
            std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr)
        : ruleEvaluationFactoryPtr_(ruleEvaluationFactoryPtr), labelMatrixPtr_(labelMatrixPtr) {

    }

    std::unique_ptr<ILabelWiseStatistics> DenseLabelWiseStatisticsFactory::create() const {
        // The number of examples
        uint32 numExamples = labelMatrixPtr_->getNumRows();
        // The number of labels
        uint32 numLabels = labelMatrixPtr_->getNumCols();
        // A matrix that stores the weights of individual examples and labels that are still uncovered
        float64* uncoveredLabels = (float64*) malloc(numExamples * numLabels * sizeof(float64));
        // The sum of weights of all examples and labels that remain to be covered
        float64 sumUncoveredLabels = 0;
        // An array that stores whether rules should predict individual labels as relevant (1) or irrelevant (0)
        uint8* minorityLabels = (uint8*) malloc(numLabels * sizeof(uint8));
        // The number of positive examples that must be exceeded for the default rule to predict a label as relevant
        float64 threshold = numExamples / 2.0;

        for (uint32 c = 0; c < numLabels; c++) {
            uint32 numPositiveLabels = 0;

            for (uint32 r = 0; r < numExamples; r++) {
                uint8 trueLabel = labelMatrixPtr_->getValue(r, c);
                numPositiveLabels += trueLabel;

                // Mark the current example and label as uncovered...
                uncoveredLabels[r * numLabels + c] = 1;
            }

            if (numPositiveLabels > threshold) {
                minorityLabels[c] = 0;
                sumUncoveredLabels += (numExamples - numPositiveLabels);
            } else {
                minorityLabels[c] = 1;
                sumUncoveredLabels += numPositiveLabels;
            }
        }

        return std::make_unique<LabelWiseStatistics>(ruleEvaluationFactoryPtr_, labelMatrixPtr_, uncoveredLabels,
                                                     sumUncoveredLabels, minorityLabels);
    }

}
