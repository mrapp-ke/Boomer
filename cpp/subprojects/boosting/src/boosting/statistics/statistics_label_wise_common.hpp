/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/statistics/statistics_label_wise.hpp"


namespace boosting {

    template<typename Prediction, typename LabelMatrix, typename StatisticView, typename ScoreMatrix,
             typename LossFunction>
    static inline void applyLabelWisePredictionInternally(uint32 statisticIndex, const Prediction& prediction,
                                                          const LabelMatrix& labelMatrix, StatisticView& statisticView,
                                                          ScoreMatrix& scoreMatrix, const LossFunction& lossFunction) {
        // Update the scores that are currently predicted for the example at the given index...
        scoreMatrix.addToRowFromSubset(statisticIndex, prediction.scores_cbegin(), prediction.scores_cend(),
                                       prediction.indices_cbegin(), prediction.indices_cend());

        // Update the gradients and Hessians of the example at the given index...
        lossFunction.updateLabelWiseStatistics(statisticIndex, labelMatrix, scoreMatrix, prediction.indices_cbegin(),
                                               prediction.indices_cend(), statisticView);
    }

    /**
     * An abstract base class for all statistics that provide access to gradients and Hessians that are calculated
     * according to a differentiable loss function that is applied label-wise.
     *
     * @tparam StatisticVector          The type of the vectors that are used to store gradients and Hessians
     * @tparam StatisticView            The type of the view that provides access to the gradients and Hessians
     * @tparam ScoreMatrix              The type of the matrices that are used to store predicted scores
     * @tparam RuleEvaluationFactory    The type of the classes that may be used for calculating the predictions, as
     *                                  well as corresponding quality scores, of rules
     */
    template<typename StatisticVector, typename StatisticView, typename ScoreMatrix, typename RuleEvaluationFactory>
    class AbstractLabelWiseImmutableStatistics : virtual public IImmutableStatistics {

        protected:

            /**
             * An abstract base class for all subsets of the gradients and Hessians that are stored by an instance of
             * the class `AbstractLabelWiseImmutableStatistics`.
             *
             * @tparam T The type of the vector that provides access to the indices of the labels that are included in
             *           the subset
             */
            template<typename T>
            class AbstractStatisticsSubset : public IStatisticsSubset {

                private:

                    const AbstractLabelWiseImmutableStatistics& statistics_;

                    std::unique_ptr<IRuleEvaluation<StatisticVector>> ruleEvaluationPtr_;

                    const T& labelIndices_;

                    StatisticVector sumVector_;

                    StatisticVector* accumulatedSumVector_;

                    StatisticVector tmpVector_;

                protected:

                    /**
                     * A pointer to an object of template type `StatisticVector` that stores the total sum of all
                     * gradients and Hessians.
                     */
                    const StatisticVector* totalSumVector_;

                    /**
                     * Returns the view that provides access to the gradients and Hessians.
                     *
                     * @return A reference to an object of template type `StatisticView` that provides access to the
                     *         gradient and Hessians
                     */
                    const StatisticView& getStatisticView() {
                        return *statistics_.statisticViewPtr_;
                    }

                public:

                    /**
                     * @param statistics        A reference to an object of type `AbstractLabelWiseImmutableStatistics`
                     *                          that stores the gradients and Hessians
                     * @param totalSumVector    A pointer to an object of template type `StatisticVector` that stores
                     *                          the total sums of gradients and Hessians
                     * @param ruleEvaluationPtr An unique pointer to an object of type `IRuleEvaluation` that should be
                     *                          used to calculate the predictions, as well as corresponding quality
                     *                          scores, of rules
                     * @param labelIndices      A reference to an object of template type `T` that provides access to
                     *                          the indices of the labels that are included in the subset
                     */
                    AbstractStatisticsSubset(const AbstractLabelWiseImmutableStatistics& statistics,
                                             const StatisticVector* totalSumVector,
                                             std::unique_ptr<IRuleEvaluation<StatisticVector>> ruleEvaluationPtr,
                                             const T& labelIndices)
                        : statistics_(statistics), ruleEvaluationPtr_(std::move(ruleEvaluationPtr)),
                          labelIndices_(labelIndices), sumVector_(StatisticVector(labelIndices.getNumElements(), true)),
                          accumulatedSumVector_(nullptr), tmpVector_(StatisticVector(labelIndices.getNumElements())),
                          totalSumVector_(totalSumVector) {

                    }

                    ~AbstractStatisticsSubset() override {
                        delete accumulatedSumVector_;
                    }

                    /**
                     * @see `IStatisticsSubset::addToSubset`
                     */
                    void addToSubset(uint32 statisticIndex, float64 weight) override final {
                        sumVector_.addToSubset(*statistics_.statisticViewPtr_, statisticIndex, labelIndices_, weight);
                    }

                    /**
                     * @see `IStatisticsSubset::resetSubset`
                     */
                    void resetSubset() override final {
                        if (!accumulatedSumVector_) {
                            // Create a vector for storing the accumulated sums of gradients and Hessians, if
                            // necessary...
                            accumulatedSumVector_ = new StatisticVector(sumVector_);
                        } else {
                            // Add the sums of gradients and Hessians to the accumulated sums of gradients and
                            // Hessians...
                            accumulatedSumVector_->add(sumVector_);
                        }

                        // Reset the sums of gradients and Hessians to zero...
                        sumVector_.clear();
                    }

                    /**
                     * @see `IStatisticsSubset::calculatePrediction`
                     */
                    const IScoreVector& calculatePrediction(bool uncovered, bool accumulated) override final {
                        StatisticVector& sumsOfStatistics = accumulated ? *accumulatedSumVector_ : sumVector_;

                        if (uncovered) {
                            tmpVector_.difference(*totalSumVector_, labelIndices_, sumsOfStatistics);
                            return ruleEvaluationPtr_->calculatePrediction(tmpVector_);
                        }

                        return ruleEvaluationPtr_->calculatePrediction(sumsOfStatistics);
                    }

            };

        private:

            uint32 numStatistics_;

            uint32 numLabels_;

        protected:

            /**
             * An unique pointer to an object of template type `StatisticView` that stores the gradients and Hessians.
             */
            std::unique_ptr<StatisticView> statisticViewPtr_;

            /**
             * A pointer to an object of template type `RuleEvaluationFactory` to be used for calculating the
             * predictions, as well as corresponding quality scores, of rules.
             */
            const RuleEvaluationFactory* ruleEvaluationFactoryPtr_;

        public:

            /**
             * @param statisticViewPtr      An unique pointer to an object of template type `StatisticView` that
             *                              provides access to the the gradients and Hessians
             * @param ruleEvaluationFactory A reference to an object of template type `RuleEvaluationFactory`, that
             *                              allows to create instances of the class that is used for calculating the
             *                              predictions, as well as corresponding quality scores, of rules
             */
            AbstractLabelWiseImmutableStatistics(std::unique_ptr<StatisticView> statisticViewPtr,
                                                 const RuleEvaluationFactory& ruleEvaluationFactory)
                : numStatistics_(statisticViewPtr->getNumRows()), numLabels_(statisticViewPtr->getNumCols()),
                  statisticViewPtr_(std::move(statisticViewPtr)), ruleEvaluationFactoryPtr_(&ruleEvaluationFactory) {

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

    };

    /**
     * Provides access to gradients and Hessians that are calculated according to a differentiable loss function that is
     * applied label-wise and are organized as a histogram.
     *
     * @tparam StatisticVector          The type of the vectors that are used to store gradients and Hessians
     * @tparam StatisticView            The type of the view that provides access to the original gradients and Hessians
     * @tparam Histogram                The type of a histogram that stores aggregated gradients and Hessians
     * @tparam ScoreMatrix              The type of the matrices that are used to store predicted scores
     * @tparam RuleEvaluationFactory    The type of the classes that may be used for calculating the predictions, as
     *                                  well as corresponding quality scores, of rules
     */
    template<typename StatisticVector, typename StatisticView, typename Histogram, typename ScoreMatrix,
             typename RuleEvaluationFactory>
    class LabelWiseHistogram final : public AbstractLabelWiseImmutableStatistics<StatisticVector, Histogram,
                                                                                 ScoreMatrix, RuleEvaluationFactory>,
                                     virtual public IHistogram {

        private:

            /**
             * Provides access to a subset of the gradients and Hessians that are stored by an instance of the class
             * `LabelWiseHistogram`.
             *
             * @tparam T The type of the vector that provides access to the indices of the labels that are included in
             *           the subset
             */
            template<typename T>
            class StatisticsSubset final :
                    public AbstractLabelWiseImmutableStatistics<StatisticVector, Histogram, ScoreMatrix,
                                                                RuleEvaluationFactory>::template AbstractStatisticsSubset<T> {

                private:

                    const LabelWiseHistogram& histogram_;

                    StatisticVector* totalCoverableSumVector_;

                public:

                    /**
                     * @param histogram         A reference to an object of type `LabelWiseHistogram` that stores the
                     *                          gradients and Hessians
                     * @param totalSumVector    A pointer to an object of template type `StatisticVector` that stores
                     *                          the total sums of gradients and Hessians
                     * @param ruleEvaluationPtr An unique pointer to an object of type `IRuleEvaluation` that should be
                     *                          used to calculate the predictions, as well as corresponding quality
                     *                          scores, of rules
                     * @param labelIndices      A reference to an object of template type `T` that provides access to
                     *                          the indices of the labels that are included in the subset
                     */
                    StatisticsSubset(const LabelWiseHistogram& histogram, const StatisticVector* totalSumVector,
                            std::unique_ptr<IRuleEvaluation<StatisticVector>> ruleEvaluationPtr, const T& labelIndices)
                        : AbstractLabelWiseImmutableStatistics<StatisticVector, Histogram, ScoreMatrix,
                                                               RuleEvaluationFactory>::template AbstractStatisticsSubset<T>(
                              histogram, totalSumVector, std::move(ruleEvaluationPtr), labelIndices),
                          histogram_(histogram), totalCoverableSumVector_(nullptr) {

                    }

                    ~StatisticsSubset() override {
                        delete totalCoverableSumVector_;
                    }

                    /**
                     * @see `IStatisticsSubset::addToMissing`
                     */
                    void addToMissing(uint32 statisticIndex, float64 weight) override {
                        // Create a vector for storing the totals sums of gradients and Hessians, if necessary...
                        if (!totalCoverableSumVector_) {
                            totalCoverableSumVector_ = new StatisticVector(*this->totalSumVector_);
                            this->totalSumVector_ = totalCoverableSumVector_;
                        }

                        // Subtract the gradients and Hessians of the example at the given index (weighted by the given
                        // weight) from the total sums of gradients and Hessians...
                        const StatisticView& originalStatisticView = histogram_.originalStatisticView_;
                        totalCoverableSumVector_->add(originalStatisticView, statisticIndex, -weight);
                    }

            };

            const StatisticView& originalStatisticView_;

            const StatisticVector* totalSumVector_;

        public:

            /**
             * @param originalStatisticView     A reference to an object of template type `StatisticView` that provides
             *                                  access to the original gradients and Hessians, the histogram was created
             *                                  from
             * @param totalSumVector            A pointer to an object of template type `StatisticVector` that stores
             *                                  the total sums of gradients and Hessians
             * @param ruleEvaluationFactory     A reference to an object of type `RuleEvaluationFactory`, that
             *                                  allows to create instances of the class that is used for calculating the
             *                                  predictions, as well as corresponding quality scores, of rules
             * @param numBins                   The number of bins in the histogram
             */
            LabelWiseHistogram(const StatisticView& originalStatisticView, const StatisticVector* totalSumVector,
                               const RuleEvaluationFactory& ruleEvaluationFactory, uint32 numBins)
                : AbstractLabelWiseImmutableStatistics<StatisticVector, Histogram, ScoreMatrix, RuleEvaluationFactory>(
                      std::make_unique<Histogram>(numBins, originalStatisticView.getNumCols()), ruleEvaluationFactory),
                  originalStatisticView_(originalStatisticView), totalSumVector_(totalSumVector) {

            }

            /**
             * @see `IHistogram::clear`
             */
            void clear() override {
                this->statisticViewPtr_->clear();
            }

            /**
             * @see `IHistogram::addToBin`
             */
            void addToBin(uint32 binIndex, uint32 statisticIndex, float64 weight) override {
                this->statisticViewPtr_->addToRow(binIndex, originalStatisticView_.row_cbegin(statisticIndex),
                                                  originalStatisticView_.row_cend(statisticIndex), weight);
            }

            /**
             * @see `IImmutableStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const CompleteIndexVector& labelIndices) const override {
                std::unique_ptr<IRuleEvaluation<StatisticVector>> ruleEvaluationPtr =
                    this->ruleEvaluationFactoryPtr_->create(*totalSumVector_, labelIndices);
                return std::make_unique<StatisticsSubset<CompleteIndexVector>>(*this, totalSumVector_,
                                                                               std::move(ruleEvaluationPtr),
                                                                               labelIndices);
            }

            /**
             * @see `IImmutableStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const PartialIndexVector& labelIndices) const override {
                std::unique_ptr<IRuleEvaluation<StatisticVector>> ruleEvaluationPtr =
                    this->ruleEvaluationFactoryPtr_->create(*totalSumVector_, labelIndices);
                return std::make_unique<StatisticsSubset<PartialIndexVector>>(*this, totalSumVector_,
                                                                              std::move(ruleEvaluationPtr),
                                                                              labelIndices);
            }

    };

    /**
     * An abstract base class for all classes that provide access to gradients and Hessians that are calculated
     * according to a differentiable loss function that is applied label-wise and allows to update the gradients and
     * Hessians after a new rule has been learned.
     *
     * @tparam LabelMatrix              The type of the matrix that provides access to the labels of the training
     *                                  examples
     * @tparam StatisticVector          The type of the vectors that are used to store gradients and Hessians
     * @tparam StatisticView            The type of the view that provides access to the gradients and Hessians
     * @tparam Histogram                The type of a histogram that stores aggregated gradients and Hessians
     * @tparam ScoreMatrix              The type of the matrices that are used to store predicted scores
     * @tparam LossFunction             The type of the loss function that is used to calculate gradients and Hessians
     * @tparam EvaluationMeasure        The type of the evaluation measure that is used to assess the quality of
     *                                  predictions for a specific statistic
     * @tparam RuleEvaluationFactory    The type of the classes that may be used for calculating the predictions, as
     *                                  well as corresponding quality scores, of rules
     */
    template<typename LabelMatrix, typename StatisticVector, typename StatisticView, typename Histogram,
             typename ScoreMatrix, typename LossFunction, typename EvaluationMeasure, typename RuleEvaluationFactory>
    class AbstractLabelWiseStatistics : public AbstractLabelWiseImmutableStatistics<StatisticVector, StatisticView,
                                                                                    ScoreMatrix, RuleEvaluationFactory>,
                                        virtual public ILabelWiseStatistics<RuleEvaluationFactory> {

        private:

            /**
             * Provides access to a subset of the gradients and Hessians that are stored by an instance of the class
             * `AbstractLabelWiseStatistics`.
             *
             * @tparam T The type of the vector that provides access to the indices of the labels that are included in
             *           the subset
             */
            template<typename T>
            class StatisticsSubset final :
                    public AbstractLabelWiseImmutableStatistics<StatisticVector, StatisticView, ScoreMatrix,
                                                                RuleEvaluationFactory>::template AbstractStatisticsSubset<T> {

                private:

                    StatisticVector* totalCoverableSumVector_;

                public:

                    /**
                     * @param statistics        A reference to an object of type `AbstractLabelWiseStatistics` that
                     *                          stores the gradients and Hessians
                     * @param totalSumVector    A pointer to an object of template type `StatisticVector` that stores
                     *                          the total sums of gradients and Hessians
                     * @param ruleEvaluationPtr An unique pointer to an object of type `IRuleEvaluation` that should be
                     *                          used to calculate the predictions, as well as corresponding quality
                     *                          scores, of rules
                     * @param labelIndices      A reference to an object of template type `T` that provides access to
                     *                          the indices of the labels that are included in the subset
                     */
                    StatisticsSubset(const AbstractLabelWiseStatistics& statistics,
                                     const StatisticVector* totalSumVector,
                                     std::unique_ptr<IRuleEvaluation<StatisticVector>> ruleEvaluationPtr,
                                     const T& labelIndices)
                        : AbstractLabelWiseImmutableStatistics<StatisticVector, StatisticView, ScoreMatrix,
                                                               RuleEvaluationFactory>::template AbstractStatisticsSubset<T>(
                              statistics, totalSumVector, std::move(ruleEvaluationPtr), labelIndices),
                          totalCoverableSumVector_(nullptr) {

                    }

                    ~StatisticsSubset() override {
                        delete totalCoverableSumVector_;
                    }

                    /**
                     * @see `IStatisticsSubset::addToMissing`
                     */
                    void addToMissing(uint32 statisticIndex, float64 weight) override {
                        // Create a vector for storing the totals sums of gradients and Hessians, if necessary...
                        if (!totalCoverableSumVector_) {
                            totalCoverableSumVector_ = new StatisticVector(*this->totalSumVector_);
                            this->totalSumVector_ = totalCoverableSumVector_;
                        }

                        // Subtract the gradients and Hessians of the example at the given index (weighted by the given
                        // weight) from the total sums of gradients and Hessians...
                        const StatisticView& statisticView = this->getStatisticView();
                        totalCoverableSumVector_->add(statisticView, statisticIndex, -weight);
                    }

            };

            std::unique_ptr<StatisticVector> totalSumVectorPtr_;

            const std::unique_ptr<LossFunction> lossPtr_;

            const std::unique_ptr<EvaluationMeasure> evaluationMeasurePtr_;

            const LabelMatrix& labelMatrix_;

            std::unique_ptr<ScoreMatrix> scoreMatrixPtr_;

        public:

            /**
             * @param lossPtr               An unique pointer to an object of template type `LossFunction` that
             *                              implements the loss function that should be used for calculating gradients
             *                              and Hessians
             * @param evaluationMeasurePtr  An unique pointer to an object of template type `EvaluationMeasure` that
             *                              implements the evaluation measure that should be used to assess the quality
             *                              of predictions for a specific statistic
             * @param ruleEvaluationFactory A reference to an object of type `RuleEvaluationFactory`, that allows to
             *                              create instances of the class that is used for calculating the
             *                              predictions, as well as corresponding quality scores, of rules
             * @param labelMatrix           A reference to an object of template type `LabelMatrix` that provides access
             *                              to the labels of the training examples
             * @param statisticViewPtr      An unique pointer to an object of template type `StatisticView` that
             *                              provides access to the gradients and Hessians
             * @param scoreMatrixPtr        An unique pointer to an object of template type `ScoreMatrix` that stores
             *                              the currently predicted scores
             */
            AbstractLabelWiseStatistics(std::unique_ptr<LossFunction> lossPtr,
                                        std::unique_ptr<EvaluationMeasure> evaluationMeasurePtr,
                                        const RuleEvaluationFactory& ruleEvaluationFactory,
                                        const LabelMatrix& labelMatrix, std::unique_ptr<StatisticView> statisticViewPtr,
                                        std::unique_ptr<ScoreMatrix> scoreMatrixPtr)
                : AbstractLabelWiseImmutableStatistics<StatisticVector, StatisticView, ScoreMatrix,
                                                       RuleEvaluationFactory>(
                      std::move(statisticViewPtr), ruleEvaluationFactory),
                  totalSumVectorPtr_(std::make_unique<StatisticVector>(this->statisticViewPtr_->getNumCols())),
                  lossPtr_(std::move(lossPtr)), evaluationMeasurePtr_(std::move(evaluationMeasurePtr)),
                  labelMatrix_(labelMatrix), scoreMatrixPtr_(std::move(scoreMatrixPtr)) {

            }

            /**
             * @see `ILabelWiseStatistics::setRuleEvaluationFactory`
             */
            void setRuleEvaluationFactory(const RuleEvaluationFactory& ruleEvaluationFactory) override final {
                this->ruleEvaluationFactoryPtr_ = &ruleEvaluationFactory;
            }

            /**
             * @see `IStatistics::resetSampledStatistics`
             */
            void resetSampledStatistics() override final {
                // This function is equivalent to the function `resetCoveredStatistics`...
                this->resetCoveredStatistics();
            }

            /**
             * @see `IStatistics::addSampledStatistic`
             */
            void addSampledStatistic(uint32 statisticIndex, float64 weight) override final {
                // This function is equivalent to the function `updateCoveredStatistic`...
                this->updateCoveredStatistic(statisticIndex, weight, false);
            }

            /**
             * @see `IStatistics::resetCoveredStatistic`
             */
            void resetCoveredStatistics() override final {
                totalSumVectorPtr_->clear();
            }

            /**
             * @see `IStatistics::updateCoveredStatistic`
             */
            void updateCoveredStatistic(uint32 statisticIndex, float64 weight, bool remove) override final {
                float64 signedWeight = remove ? -weight : weight;
                totalSumVectorPtr_->add(*this->statisticViewPtr_, statisticIndex, signedWeight);
            }

            /**
             * @see `IStatistics::applyPrediction`
             */
            void applyPrediction(uint32 statisticIndex, const CompletePrediction& prediction) override final {
                applyLabelWisePredictionInternally<CompletePrediction, LabelMatrix, StatisticView, ScoreMatrix,
                                                   LossFunction>(
                    statisticIndex, prediction, labelMatrix_, *this->statisticViewPtr_, *scoreMatrixPtr_, *lossPtr_);
            }

            /**
             * @see `IStatistics::applyPrediction`
             */
            void applyPrediction(uint32 statisticIndex, const PartialPrediction& prediction) override final {
                applyLabelWisePredictionInternally<PartialPrediction, LabelMatrix, StatisticView, ScoreMatrix,
                                                   LossFunction>(
                    statisticIndex, prediction, labelMatrix_, *this->statisticViewPtr_, *scoreMatrixPtr_, *lossPtr_);
            }

            /**
             * @see `IStatistics::evaluatePrediction`
             */
            float64 evaluatePrediction(uint32 statisticIndex) const override final {
                return evaluationMeasurePtr_->evaluate(statisticIndex, labelMatrix_, *scoreMatrixPtr_);
            }

            /**
             * @see `IStatistics::createHistogram`
             */
            std::unique_ptr<IHistogram> createHistogram(uint32 numBins) const override final {
                return std::make_unique<LabelWiseHistogram<StatisticVector, StatisticView, Histogram, ScoreMatrix,
                                                           RuleEvaluationFactory>>(
                    *this->statisticViewPtr_, totalSumVectorPtr_.get(), *this->ruleEvaluationFactoryPtr_, numBins);
            }

            /**
             * @see `IImmutableStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
                    const CompleteIndexVector& labelIndices) const override final {
                std::unique_ptr<IRuleEvaluation<StatisticVector>> ruleEvaluationPtr =
                    this->ruleEvaluationFactoryPtr_->create(*totalSumVectorPtr_, labelIndices);
                return std::make_unique<StatisticsSubset<CompleteIndexVector>>(*this, totalSumVectorPtr_.get(),
                                                                               std::move(ruleEvaluationPtr),
                                                                               labelIndices);
            }

            /**
             * @see `IImmutableStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
                    const PartialIndexVector& labelIndices) const override final {
                std::unique_ptr<IRuleEvaluation<StatisticVector>> ruleEvaluationPtr =
                    this->ruleEvaluationFactoryPtr_->create(*totalSumVectorPtr_, labelIndices);
                return std::make_unique<StatisticsSubset<PartialIndexVector>>(*this, totalSumVectorPtr_.get(),
                                                                              std::move(ruleEvaluationPtr),
                                                                              labelIndices);
            }

    };

}
