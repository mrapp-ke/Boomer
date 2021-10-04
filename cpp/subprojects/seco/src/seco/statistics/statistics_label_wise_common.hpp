/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#include "seco/statistics/statistics_label_wise.hpp"


namespace seco {

    template<typename Prediction, typename WeightMatrix>
    static inline void applyLabelWisePredictionInternally(uint32 statisticIndex, const Prediction& prediction,
                                                          WeightMatrix& weightMatrix,
                                                          const BinarySparseArrayVector& majorityLabelVector) {
        weightMatrix.updateRow(statisticIndex, majorityLabelVector, prediction.scores_cbegin(),
                               prediction.scores_cend(), prediction.indices_cbegin(), prediction.indices_cend());
    }

    /**
     * An abstract base class for all statistics that provide access to the elements of confusion matrices that are
     * computed independently for each label.
     *
     * @tparam LabelMatrix              The type of the matrix that provides access to the labels of the training
     *                                  examples
     * @tparam WeightMatrix             The type of the matrix that is used to store the weights of individual examples
     *                                  and labels
     * @tparam ConfusionMatrixVector    The type of the vector that is used to store confusion matrices
     * @tparam RuleEvaluationFactory    The type of the classes that may be used for calculating the predictions, as
     *                                  well as corresponding quality scores, of rules
     */
    template<typename LabelMatrix, typename WeightMatrix, typename ConfusionMatrixVector,
             typename RuleEvaluationFactory>
    class AbstractLabelWiseStatistics : public ILabelWiseStatistics<RuleEvaluationFactory> {

        private:

            /**
             * Provides access to a subset of the confusion matrices that are stored by an instance of the class
             * `AbstractLabelWiseStatistics`.
             *
             * @tparam T The type of the vector that provides access to the indices of the labels that are included in
             *           the subset
             */
            template<typename T>
            class StatisticsSubset final : public IStatisticsSubset {

                private:

                    const AbstractLabelWiseStatistics& statistics_;

                    const ConfusionMatrixVector* totalSumVector_;

                    std::unique_ptr<IRuleEvaluation> ruleEvaluationPtr_;

                    const T& labelIndices_;

                    ConfusionMatrixVector sumVector_;

                    ConfusionMatrixVector* accumulatedSumVector_;

                    ConfusionMatrixVector* totalCoverableSumVector_;

                    ConfusionMatrixVector tmpVector_;

                public:

                    /**
                     * @param statistics        A reference to an object of type `AbstractLabelWiseStatistics` that
                     *                          stores the confusion matrices
                     * @param ruleEvaluationPtr An unique pointer to an object of type `IRuleEvaluation` that should be
                     *                          used to calculate the predictions, as well as corresponding quality
                     *                          scores, of rules
                     * @param labelIndices      A reference to an object of template type `T` that provides access to
                     *                          the indices of the labels that are included in the subset
                     */
                    StatisticsSubset(const AbstractLabelWiseStatistics& statistics,
                                     std::unique_ptr<IRuleEvaluation> ruleEvaluationPtr, const T& labelIndices)
                        : statistics_(statistics), totalSumVector_(&statistics_.subsetSumVector_),
                          ruleEvaluationPtr_(std::move(ruleEvaluationPtr)), labelIndices_(labelIndices),
                          sumVector_(ConfusionMatrixVector(labelIndices.getNumElements(), true)),
                          accumulatedSumVector_(nullptr), totalCoverableSumVector_(nullptr),
                          tmpVector_(ConfusionMatrixVector(labelIndices.getNumElements())) {

                    }

                    ~StatisticsSubset() {
                        delete accumulatedSumVector_;
                        delete totalCoverableSumVector_;
                    }

                    void addToMissing(uint32 statisticIndex, float64 weight) override {
                        // Allocate a vector for storing the totals sums of confusion matrices, if necessary...
                        if (totalCoverableSumVector_ == nullptr) {
                            totalCoverableSumVector_ = new ConfusionMatrixVector(*totalSumVector_);
                            totalSumVector_ = totalCoverableSumVector_;
                        }

                        // For each label, subtract the confusion matrices of the example at the given index (weighted
                        // by the given weight) from the total sum of confusion matrices...
                        totalCoverableSumVector_->add(statisticIndex, statistics_.labelMatrix_,
                                                      *statistics_.majorityLabelVectorPtr_,
                                                      *statistics_.weightMatrixPtr_, -weight);
                    }

                    void addToSubset(uint32 statisticIndex, float64 weight) override {
                        sumVector_.addToSubset(statisticIndex, statistics_.labelMatrix_,
                                               *statistics_.majorityLabelVectorPtr_, *statistics_.weightMatrixPtr_,
                                               labelIndices_, weight);
                    }

                    void resetSubset() override {
                        // Allocate a vector for storing the accumulated confusion matrices, if necessary...
                        if (accumulatedSumVector_ == nullptr) {
                            uint32 numPredictions = labelIndices_.getNumElements();
                            accumulatedSumVector_ = new ConfusionMatrixVector(numPredictions, true);
                        }

                        // Reset the confusion matrix for each label to zero and add its elements to the accumulated
                        // confusion matrix...
                        accumulatedSumVector_->add(sumVector_.cbegin(), sumVector_.cend());
                        sumVector_.clear();
                    }

                    const IScoreVector& calculatePrediction(bool uncovered, bool accumulated) override {
                        const ConfusionMatrixVector& sumsOfConfusionMatrices =
                            accumulated ? *accumulatedSumVector_ : sumVector_;

                        if (uncovered) {
                            tmpVector_.difference(totalSumVector_->cbegin(), totalSumVector_->cend(), labelIndices_,
                                                  sumsOfConfusionMatrices.cbegin(), sumsOfConfusionMatrices.cend());
                            return ruleEvaluationPtr_->calculatePrediction(*statistics_.majorityLabelVectorPtr_,
                                                                           statistics_.totalSumVector_, tmpVector_);
                        }

                        return ruleEvaluationPtr_->calculatePrediction(*statistics_.majorityLabelVectorPtr_,
                                                                       statistics_.totalSumVector_,
                                                                       sumsOfConfusionMatrices);
                    }

            };

            uint32 numStatistics_;

            uint32 numLabels_;

            const RuleEvaluationFactory* ruleEvaluationFactoryPtr_;

            const LabelMatrix& labelMatrix_;

            std::unique_ptr<BinarySparseArrayVector> majorityLabelVectorPtr_;

            ConfusionMatrixVector totalSumVector_;

            ConfusionMatrixVector subsetSumVector_;

        protected:

            /**
             * An unique pointer to an object of template type `WeightMatrix` that stores the weights of individual
             * examples and labels.
             */
            std::unique_ptr<WeightMatrix> weightMatrixPtr_;

        public:

            /**
             * @param ruleEvaluationFactory     A reference to an object of template type `RuleEvaluationFactory` that
             *                                  allows to create instances of the class that is used for calculating the
             *                                  predictions, as well as corresponding quality scores, of rules
             * @param labelMatrix               A reference to an object of template type `LabelMatrix` that provides
             *                                  access to the labels of the training examples
             * @param weightMatrixPtr           An unique pointer to an object of template type `WeightMatrix` that
             *                                  stores the weights of individual examples and labels
             * @param majorityLabelVectorPtr    An unique pointer to an object of type `BinarySparseArrayVector` that
             *                                  stores the predictions of the default rule
             */
            AbstractLabelWiseStatistics(const RuleEvaluationFactory& ruleEvaluationFactory,
                                        const LabelMatrix& labelMatrix, std::unique_ptr<WeightMatrix> weightMatrixPtr,
                                        std::unique_ptr<BinarySparseArrayVector> majorityLabelVectorPtr)
                : numStatistics_(labelMatrix.getNumRows()), numLabels_(labelMatrix.getNumCols()),
                  ruleEvaluationFactoryPtr_(&ruleEvaluationFactory), labelMatrix_(labelMatrix),
                  majorityLabelVectorPtr_(std::move(majorityLabelVectorPtr)),
                  totalSumVector_(ConfusionMatrixVector(numLabels_)),
                  subsetSumVector_(ConfusionMatrixVector(numLabels_)), weightMatrixPtr_(std::move(weightMatrixPtr)) {

            }

            /**
             * @see `IImmutableStatistics::getNumStatistics`
             */
            uint32 getNumStatistics() const override final {
                return numStatistics_;
            }

            /**
             * @see `IImmutableStatistics::getNumLabels`
             */
            uint32 getNumLabels() const override final {
                return numLabels_;
            }

            /**
             * @see `ICoverageStatistics::getSumOfUncoveredWeights`
             */
            float64 getSumOfUncoveredWeights() const override final {
                return weightMatrixPtr_->getSumOfUncoveredWeights();
            }

            /**
             * @see `ILabelWiseStatistics::setRuleEvaluationFactory`
             */
            void setRuleEvaluationFactory(const RuleEvaluationFactory& ruleEvaluationFactory) override final {
                ruleEvaluationFactoryPtr_ = &ruleEvaluationFactory;
            }

            /**
             * @see `IStatistics::resetSampledStatistics`
             */
            void resetSampledStatistics() override final {
                totalSumVector_.clear();
                subsetSumVector_.clear();
            }

            /**
             * @see `IStatistics::addSampledStatistic`
             */
            void addSampledStatistic(uint32 statisticIndex, float64 weight) override final {
                totalSumVector_.add(statisticIndex, labelMatrix_, *majorityLabelVectorPtr_, *weightMatrixPtr_, weight);
                subsetSumVector_.add(statisticIndex, labelMatrix_, *majorityLabelVectorPtr_, *weightMatrixPtr_, weight);
            }

            /**
             * @see `IStatistics::resetCoveredStatistics`
             */
            void resetCoveredStatistics() override final {
                subsetSumVector_.clear();
            }

            /**
             * @see `IStatistics::updateCoveredStatistic`
             */
            void updateCoveredStatistic(uint32 statisticIndex, float64 weight, bool remove) override final {
                float64 signedWeight = remove ? -weight : weight;
                subsetSumVector_.add(statisticIndex, labelMatrix_, *majorityLabelVectorPtr_, *weightMatrixPtr_,
                                     signedWeight);
            }

            /**
             * @see `IImmutableStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
                    const CompleteIndexVector& labelIndices) const override final {
                std::unique_ptr<IRuleEvaluation> ruleEvaluationPtr = ruleEvaluationFactoryPtr_->create(labelIndices);
                return std::make_unique<StatisticsSubset<CompleteIndexVector>>(*this, std::move(ruleEvaluationPtr),
                                                                               labelIndices);
            }

            /**
             * @see `IImmutableStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
                    const PartialIndexVector& labelIndices) const override final {
                std::unique_ptr<IRuleEvaluation> ruleEvaluationPtr = ruleEvaluationFactoryPtr_->create(labelIndices);
                return std::make_unique<StatisticsSubset<PartialIndexVector>>(*this, std::move(ruleEvaluationPtr),
                                                                              labelIndices);
            }

            /**
             * @see `IStatistics::applyPrediction`
             */
            void applyPrediction(uint32 statisticIndex, const CompletePrediction& prediction) override final {
                applyLabelWisePredictionInternally<CompletePrediction, WeightMatrix>(statisticIndex, prediction,
                                                                                     *weightMatrixPtr_,
                                                                                     *majorityLabelVectorPtr_);
            }

            /**
             * @see `IStatistics::applyPrediction`
             */
            void applyPrediction(uint32 statisticIndex, const PartialPrediction& prediction) override final {
                applyLabelWisePredictionInternally<PartialPrediction, WeightMatrix>(statisticIndex, prediction,
                                                                                    *weightMatrixPtr_,
                                                                                    *majorityLabelVectorPtr_);
            }

            /**
             * @see `IStatistics::evaluatePrediction`
             */
            float64 evaluatePrediction(uint32 statisticIndex) const override final {
                // TODO Support evaluation of predictions
                return 0;
            }

            /**
             * @see `IStatistics::createHistogram`
             */
            std::unique_ptr<IHistogram> createHistogram(uint32 numBins) const override final {
                //TODO Support creation of histograms
                return nullptr;
            }

    };

}
