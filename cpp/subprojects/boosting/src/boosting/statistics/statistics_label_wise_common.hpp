#include "boosting/statistics/statistics_label_wise.hpp"
#include "common/statistics/statistics_subset_decomposable.hpp"


namespace boosting {

    /**
     * An abstract base class for all statistics that provide access to gradients and Hessians that are calculated
     * according to a differentiable loss function that is applied label-wise.
     *
     * @tparam StatisticVector  The type of the vectors that are used to store gradients and Hessians
     * @tparam StatisticMatrix  The type of the matrices that are used to store gradients and Hessians
     * @tparam ScoreMatrix      The type of the matrices that are used to store predicted scores
     */
    template<class StatisticVector, class StatisticMatrix, class ScoreMatrix>
    class AbstractLabelWiseStatistics : virtual public IImmutableStatistics {

        protected:

            /**
             * Provides access to a subset of the gradients and Hessians that are stored by an instance of the class
             * `AbstractLabelWiseStatistics`.
             *
             * @tparam T The type of the vector that provides access to the indices of the labels that are included in
             *           the subset
             */
            template<class T>
            class StatisticsSubset final : public AbstractDecomposableStatisticsSubset {

                private:

                    const AbstractLabelWiseStatistics& statistics_;

                    const StatisticVector* totalSumVector_;

                    std::unique_ptr<ILabelWiseRuleEvaluation> ruleEvaluationPtr_;

                    const T& labelIndices_;

                    StatisticVector sumVector_;

                    StatisticVector* accumulatedSumVector_;

                    StatisticVector* totalCoverableSumVector_;

                    StatisticVector tmpVector_;

                public:

                    /**
                     * @param statistics        A reference to an object of type `AbstractLabelWiseStatistics` that
                     *                          stores the gradients and Hessians
                     * @param totalSumVector    A pointer to an object of template type `StatisticVector` that stores
                     *                          the total sums of gradients and Hessians
                     * @param ruleEvaluationPtr An unique pointer to an object of type `ILabelWiseRuleEvaluation` that
                     *                          should be used to calculate the predictions, as well as corresponding
                     *                          quality scores, of rules
                     * @param labelIndices      A reference to an object of template type `T` that provides access to
                     *                          the indices of the labels that are included in the subset
                     */
                    StatisticsSubset(const AbstractLabelWiseStatistics& statistics,
                                     const StatisticVector* totalSumVector,
                                     std::unique_ptr<ILabelWiseRuleEvaluation> ruleEvaluationPtr, const T& labelIndices)
                        : statistics_(statistics), totalSumVector_(totalSumVector),
                          ruleEvaluationPtr_(std::move(ruleEvaluationPtr)), labelIndices_(labelIndices),
                          sumVector_(StatisticVector(labelIndices.getNumElements(), true)),
                          accumulatedSumVector_(nullptr), totalCoverableSumVector_(nullptr),
                          tmpVector_(StatisticVector(labelIndices.getNumElements())) {

                    }

                    ~StatisticsSubset() {
                        delete accumulatedSumVector_;
                        delete totalCoverableSumVector_;
                    }

                    void addToMissing(uint32 statisticIndex, uint32 weight) override {
                        // Create a vector for storing the totals sums of gradients and Hessians, if necessary...
                        if (totalCoverableSumVector_ == nullptr) {
                            totalCoverableSumVector_ = new StatisticVector(*totalSumVector_);
                            totalSumVector_ = totalCoverableSumVector_;
                        }

                        // Subtract the gradients and Hessians of the example at the given index (weighted by the given
                        // weight) from the total sums of gradients and Hessians...
                        totalCoverableSumVector_->subtract(
                            statistics_.statisticMatrixPtr_->gradients_row_cbegin(statisticIndex),
                            statistics_.statisticMatrixPtr_->gradients_row_cend(statisticIndex),
                            statistics_.statisticMatrixPtr_->hessians_row_cbegin(statisticIndex),
                            statistics_.statisticMatrixPtr_->hessians_row_cend(statisticIndex), weight);
                    }

                    void addToSubset(uint32 statisticIndex, uint32 weight) override {
                        sumVector_.addToSubset(statistics_.statisticMatrixPtr_->gradients_row_cbegin(statisticIndex),
                                               statistics_.statisticMatrixPtr_->gradients_row_cend(statisticIndex),
                                               statistics_.statisticMatrixPtr_->hessians_row_cbegin(statisticIndex),
                                               statistics_.statisticMatrixPtr_->hessians_row_cend(statisticIndex),
                                               labelIndices_, weight);
                    }

                    void resetSubset() override {
                        // Create a vector for storing the accumulated sums of gradients and Hessians, if necessary...
                        if (accumulatedSumVector_ == nullptr) {
                            uint32 numPredictions = labelIndices_.getNumElements();
                            accumulatedSumVector_ = new StatisticVector(numPredictions, true);
                        }

                        // Reset the sums of gradients and Hessians to zero and add it to the accumulated sums of
                        // gradients and Hessians...
                        accumulatedSumVector_->add(sumVector_.gradients_cbegin(), sumVector_.gradients_cend(),
                                                   sumVector_.hessians_cbegin(), sumVector_.hessians_cend());
                        sumVector_.setAllToZero();
                    }

                    const ILabelWiseScoreVector& calculateLabelWisePrediction(bool uncovered,
                                                                              bool accumulated) override {
                        const StatisticVector& sumsOfStatistics = accumulated ? *accumulatedSumVector_ : sumVector_;

                        if (uncovered) {
                            tmpVector_.difference(totalSumVector_->gradients_cbegin(),
                                                  totalSumVector_->gradients_cend(), totalSumVector_->hessians_cbegin(),
                                                  totalSumVector_->hessians_cend(), labelIndices_,
                                                  sumsOfStatistics.gradients_cbegin(),
                                                  sumsOfStatistics.gradients_cend(), sumsOfStatistics.hessians_cbegin(),
                                                  sumsOfStatistics.hessians_cend());
                            return ruleEvaluationPtr_->calculateLabelWisePrediction(tmpVector_);
                        }

                        return ruleEvaluationPtr_->calculateLabelWisePrediction(sumsOfStatistics);
                    }

            };

            /**
             * The type of a vector that provides access to the indices of all available labels.
             */
            typedef StatisticsSubset<FullIndexVector> FullSubset;

            /**
             * The type of a vector that provides access to the indices of a subset of the available labels.
             */
            typedef StatisticsSubset<PartialIndexVector> PartialSubset;

        private:

            uint32 numStatistics_;

            uint32 numLabels_;

        protected:

            /**
             * An unique pointer to an object of template type `StatisticMatrix` that stores the gradients and Hessians.
             */
            std::unique_ptr<StatisticMatrix> statisticMatrixPtr_;

            /**
             * A shared pointer to an object of type `IExampleWiseRuleEvaluationFactory` to be used for calculating the
             * predictions, as well as corresponding quality scores, of rules.
             */
            std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr_;

        public:

            /**
             * @param statisticMatrixPtr        An unique pointer to an object of template type `StatisticMatrix` that
             *                                  stores the gradients and Hessians
             * @param ruleEvaluationFactoryPtr  A shared pointer to an object of type `ILabelWiseRuleEvaluationFactory`,
             *                                  that allows to create instances of the class that is used for
             *                                  calculating the predictions, as well as corresponding quality scores, of
             *                                  rules
             */
            AbstractLabelWiseStatistics(std::unique_ptr<StatisticMatrix> statisticMatrixPtr,
                                        std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr)
                : numStatistics_(statisticMatrixPtr->getNumRows()), numLabels_(statisticMatrixPtr->getNumCols()),
                  statisticMatrixPtr_(std::move(statisticMatrixPtr)),
                  ruleEvaluationFactoryPtr_(ruleEvaluationFactoryPtr) {

            }

            uint32 getNumStatistics() const override final {
                return numStatistics_;
            }

            uint32 getNumLabels() const override final {
                return numLabels_;
            }

    };

    /**
     * Provides access to gradients and Hessians that are calculated according to a differentiable loss function that is
     * applied label-wise and are organized as a histogram.
     *
     * @tparam StatisticVector  The type of the vectors that are used to store gradients and Hessians
     * @tparam StatisticMatrix  The type of the matrices that are used to store gradients and Hessians
     * @tparam ScoreMatrix      The type of the matrices that are used to store predicted scores
     */
    template<class StatisticVector, class StatisticMatrix, class ScoreMatrix>
    class LabelWiseHistogram final : public AbstractLabelWiseStatistics<StatisticVector, StatisticMatrix, ScoreMatrix>,
                                     virtual public IHistogram {

        private:

            const StatisticMatrix& originalStatisticMatrix_;

            const StatisticVector* totalSumVector_;

        public:

            /**
             * @param originalStatisticMatrix   A reference to an object of template type `StatisticMatrix` that stores
             *                                  the original gradients and Hessians, the histogram was created from
             * @param totalSumVector            A pointer to an object of template type `StatisticVector` that stores
             *                                  the total sums of gradients and Hessians
             * @param ruleEvaluationFactoryPtr  A shared pointer to an object of type `ILabelWiseRuleEvaluationFactory`,
             *                                  that allows to create instances of the class that is used for
             *                                  calculating the predictions, as well as corresponding quality scores, of
             *                                  rules
             * @param numBins                   The number of bins in the histogram
             */
            LabelWiseHistogram(const StatisticMatrix& originalStatisticMatrix, const StatisticVector* totalSumVector,
                               std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr,
                               uint32 numBins)
                : AbstractLabelWiseStatistics<StatisticVector, StatisticMatrix, ScoreMatrix>(
                      std::make_unique<StatisticMatrix>(numBins, originalStatisticMatrix.getNumCols()),
                      ruleEvaluationFactoryPtr),
                  originalStatisticMatrix_(originalStatisticMatrix), totalSumVector_(totalSumVector) {

            }

            void setAllToZero() override {
                this->statisticMatrixPtr_->setAllToZero();
            }

            void addToBin(uint32 binIndex, uint32 statisticIndex, uint32 weight) override {
                this->statisticMatrixPtr_->addToRow(binIndex,
                                                    originalStatisticMatrix_.gradients_row_cbegin(statisticIndex),
                                                    originalStatisticMatrix_.gradients_row_cend(statisticIndex),
                                                    originalStatisticMatrix_.hessians_row_cbegin(statisticIndex),
                                                    originalStatisticMatrix_.hessians_row_cend(statisticIndex), weight);
            }

            std::unique_ptr<IStatisticsSubset> createSubset(const FullIndexVector& labelIndices) const override {
                std::unique_ptr<ILabelWiseRuleEvaluation> ruleEvaluationPtr =
                    this->ruleEvaluationFactoryPtr_->create(labelIndices);
                return std::make_unique<typename LabelWiseHistogram::FullSubset>(*this, totalSumVector_,
                                                                                 std::move(ruleEvaluationPtr),
                                                                                 labelIndices);
            }

            std::unique_ptr<IStatisticsSubset> createSubset(const PartialIndexVector& labelIndices) const override {
                std::unique_ptr<ILabelWiseRuleEvaluation> ruleEvaluationPtr =
                    this->ruleEvaluationFactoryPtr_->create(labelIndices);
                return std::make_unique<typename LabelWiseHistogram::PartialSubset>(*this, totalSumVector_,
                                                                                    std::move(ruleEvaluationPtr),
                                                                                    labelIndices);
            }

    };

    /**
     * Provides access to gradients and Hessians that are calculated according to a differentiable loss function that is
     * applied label-wise and allows to update the gradients and Hessians after a new rule has been learned.
     *
     * @tparam StatisticVector  The type of the vectors that are used to store gradients and Hessians
     * @tparam StatisticMatrix  The type of the matrices that are used to store gradients and Hessians
     * @tparam ScoreMatrix      The type of the matrices that are used to store predicted scores
     */
    template<class StatisticVector, class StatisticMatrix, class ScoreMatrix>
    class LabelWiseStatistics final : public AbstractLabelWiseStatistics<StatisticVector, StatisticMatrix, ScoreMatrix>,
                                      virtual public ILabelWiseStatistics {

        private:

            std::unique_ptr<StatisticVector> totalSumVectorPtr_;

            std::shared_ptr<ILabelWiseLoss> lossFunctionPtr_;

            std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr_;

            std::unique_ptr<ScoreMatrix> scoreMatrixPtr_;

            template<class T>
            void applyPredictionInternally(uint32 statisticIndex, const T& prediction) {
                // Update the scores that are currently predicted for the example at the given index...
                scoreMatrixPtr_->addToRowFromSubset(statisticIndex, prediction.scores_cbegin(),
                                                    prediction.scores_cend(), prediction.indices_cbegin(),
                                                    prediction.indices_cend());

                // Update the gradients and Hessians of the example at the given index...
                lossFunctionPtr_->updateLabelWiseStatistics(statisticIndex, *labelMatrixPtr_, *scoreMatrixPtr_,
                                                            prediction.indices_cbegin(), prediction.indices_cend(),
                                                            *this->statisticMatrixPtr_);
            }

        public:

            /**
             * @param lossFunctionPtr           A shared pointer to an object of type `ILabelWiseLoss`, representing the
             *                                  loss function to be used for calculating gradients and Hessians
             * @param ruleEvaluationFactoryPtr  A shared pointer to an object of type `ILabelWiseRuleEvaluationFactory`,
             *                                  that allows to create instances of the class that is used for
             *                                  calculating the predictions, as well as corresponding quality scores, of
             *                                  rules
             * @param labelMatrixPtr            A shared pointer to an object of type `IRandomAccessLabelMatrix` that
             *                                  provides random access to the labels of the training examples
             * @param statisticMatrixPtr        An unique pointer to an object of template type `StatisticMatrix` that
             *                                  stores the gradients and Hessians
             * @param scoreMatrixPtr            An unique pointer to an object of template type `ScoreMatrix` that
             *                                  stores the currently predicted scores
             */
            LabelWiseStatistics(std::shared_ptr<ILabelWiseLoss> lossFunctionPtr,
                                std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr,
                                std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr,
                                std::unique_ptr<StatisticMatrix> statisticMatrixPtr,
                                std::unique_ptr<ScoreMatrix> scoreMatrixPtr)
                : AbstractLabelWiseStatistics<StatisticVector, StatisticMatrix, ScoreMatrix>(
                      std::move(statisticMatrixPtr), ruleEvaluationFactoryPtr),
                  totalSumVectorPtr_(std::make_unique<StatisticVector>(this->statisticMatrixPtr_->getNumCols())),
                  lossFunctionPtr_(lossFunctionPtr), labelMatrixPtr_(labelMatrixPtr),
                  scoreMatrixPtr_(std::move(scoreMatrixPtr)) {

            }

            void setRuleEvaluationFactory(
                    std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr) override {
                this->ruleEvaluationFactoryPtr_ = ruleEvaluationFactoryPtr;
            }

            void resetSampledStatistics() override {
                // This function is equivalent to the function `resetCoveredStatistics`...
                this->resetCoveredStatistics();
            }

            void addSampledStatistic(uint32 statisticIndex, uint32 weight) override {
                // This function is equivalent to the function `updateCoveredStatistic`...
                this->updateCoveredStatistic(statisticIndex, weight, false);
            }

            void resetCoveredStatistics() override {
                totalSumVectorPtr_->setAllToZero();
            }

            void updateCoveredStatistic(uint32 statisticIndex, uint32 weight, bool remove) override {
                float64 signedWeight = remove ? -((float64) weight) : weight;
                totalSumVectorPtr_->add(this->statisticMatrixPtr_->gradients_row_cbegin(statisticIndex),
                                        this->statisticMatrixPtr_->gradients_row_cend(statisticIndex),
                                        this->statisticMatrixPtr_->hessians_row_cbegin(statisticIndex),
                                        this->statisticMatrixPtr_->hessians_row_cend(statisticIndex), signedWeight);
            }

            void applyPrediction(uint32 statisticIndex, const FullPrediction& prediction) override {
                this->applyPredictionInternally<FullPrediction>(statisticIndex, prediction);
            }

            void applyPrediction(uint32 statisticIndex, const PartialPrediction& prediction) override {
                this->applyPredictionInternally<PartialPrediction>(statisticIndex, prediction);
            }

            float64 evaluatePrediction(uint32 statisticIndex, const IEvaluationMeasure& measure) const override {
                return measure.evaluate(statisticIndex, *labelMatrixPtr_, *scoreMatrixPtr_);
            }

            std::unique_ptr<IHistogram> createHistogram(uint32 numBins) const override {
                return std::make_unique<LabelWiseHistogram<StatisticVector, StatisticMatrix, ScoreMatrix>>(
                            *this->statisticMatrixPtr_, totalSumVectorPtr_.get(), this->ruleEvaluationFactoryPtr_,
                            numBins);
            }

            std::unique_ptr<IStatisticsSubset> createSubset(const FullIndexVector& labelIndices) const override {
                std::unique_ptr<ILabelWiseRuleEvaluation> ruleEvaluationPtr =
                    this->ruleEvaluationFactoryPtr_->create(labelIndices);
                return std::make_unique<typename LabelWiseStatistics::FullSubset>(*this, totalSumVectorPtr_.get(),
                                                                                  std::move(ruleEvaluationPtr),
                                                                                  labelIndices);
            }

            std::unique_ptr<IStatisticsSubset> createSubset(const PartialIndexVector& labelIndices) const override {
                std::unique_ptr<ILabelWiseRuleEvaluation> ruleEvaluationPtr =
                    this->ruleEvaluationFactoryPtr_->create(labelIndices);
                return std::make_unique<typename LabelWiseStatistics::PartialSubset>(*this, totalSumVectorPtr_.get(),
                                                                                     std::move(ruleEvaluationPtr),
                                                                                     labelIndices);
            }

    };

}
