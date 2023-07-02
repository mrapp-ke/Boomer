/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/statistics/statistics_example_wise.hpp"
#include "common/binning/bin_weight_vector.hpp"

namespace boosting {

    static inline bool hasNonZeroWeightExampleWise(const EqualWeightVector& weights, uint32 statisticIndex) {
        return true;
    }

    template<typename WeightVector>
    static inline bool hasNonZeroWeightExampleWise(const WeightVector& weights, uint32 statisticIndex) {
        return weights[statisticIndex] != 0;
    }

    template<typename StatisticView, typename StatisticVector, typename IndexVector>
    static inline void addExampleWiseStatisticToSubset(const EqualWeightVector& weights,
                                                       const StatisticView& statisticView, StatisticVector& vector,
                                                       const IndexVector& labelIndices, uint32 statisticIndex) {
        vector.addToSubset(statisticView.gradients_cbegin(statisticIndex), statisticView.gradients_cend(statisticIndex),
                           statisticView.hessians_cbegin(statisticIndex), statisticView.hessians_cend(statisticIndex),
                           labelIndices);
    }

    template<typename WeightVector, typename StatisticView, typename StatisticVector, typename IndexVector>
    static inline void addExampleWiseStatisticToSubset(const WeightVector& weights, const StatisticView& statisticView,
                                                       StatisticVector& vector, const IndexVector& labelIndices,
                                                       uint32 statisticIndex) {
        float64 weight = weights[statisticIndex];
        vector.addToSubset(statisticView.gradients_cbegin(statisticIndex), statisticView.gradients_cend(statisticIndex),
                           statisticView.hessians_cbegin(statisticIndex), statisticView.hessians_cend(statisticIndex),
                           labelIndices, weight);
    }

    /**
     *  A subset of gradients and Hessians that are calculated according to a differentiable loss function that is
     * applied example-wise and are accessible via a view.
     *
     * @tparam StatisticVector          The type of the vector that is used to store the sums of gradients and Hessians
     * @tparam StatisticView            The type of the view that provides access to the gradients and Hessians
     * @tparam RuleEvaluationFactory    The type of the factory that allows to create instances of the class that is
     *                                  used for calculating the predictions of rules, as well as corresponding quality
     *                                  scores
     * @tparam WeightVector             The type of the vector that provides access to the weights of individual
     *                                  statistics
     * @tparam IndexVector              The type of the vector that provides access to the indices of the labels that
     *                                  are included in the subset
     */
    template<typename StatisticVector, typename StatisticView, typename RuleEvaluationFactory, typename WeightVector,
             typename IndexVector>
    class ExampleWiseStatisticsSubset : virtual public IStatisticsSubset {
        protected:

            /**
             * An object of template type `StatisticVector` that stores the sums of gradients and Hessians.
             */
            StatisticVector sumVector_;

            /**
             * A reference to an object of template type `StatisticView` that provides access to the gradients and
             * Hessians.
             */
            const StatisticView& statisticView_;

            /**
             * A reference to an object of template type `WeightVector` that provides access to the weights of
             * individual statistics.
             */
            const WeightVector& weights_;

            /**
             * A reference to an object of template type `IndexVector` that provides access to the indices of the labels
             * that are included in the subset.
             */
            const IndexVector& labelIndices_;

            /**
             * An unique pointer to an object of type `IRuleEvaluation` that is used to calculate the predictions of
             * rules, as well as their overall quality.
             */
            const std::unique_ptr<IRuleEvaluation<StatisticVector>> ruleEvaluationPtr_;

        public:

            /**
             * @param statisticView         A reference to an object of template type `StatisticView` that provides
             *                              access to the gradients and Hessians
             * @param ruleEvaluationFactory A reference to an object of template type `RuleEvaluationFactory` that
             *                              allows to create instances of the class that is used for calculating the
             *                              predictions of rules, as well as their overall quality
             * @param weights               A reference to an object of template type `WeightVector` that provides
             *                              access to the weights of individual statistics
             * @param labelIndices          A reference to an object of template type `IndexVector` that provides access
             *                              to the indices of the labels that are included in the subset
             */
            ExampleWiseStatisticsSubset(const StatisticView& statisticView,
                                        const RuleEvaluationFactory& ruleEvaluationFactory, const WeightVector& weights,
                                        const IndexVector& labelIndices)
                : sumVector_(StatisticVector(labelIndices.getNumElements(), true)), statisticView_(statisticView),
                  weights_(weights), labelIndices_(labelIndices),
                  ruleEvaluationPtr_(ruleEvaluationFactory.create(sumVector_, labelIndices)) {}

            /**
             * @see `IStatisticsSubset::hasNonZeroWeight`
             */
            bool hasNonZeroWeight(uint32 statisticIndex) const override final {
                return hasNonZeroWeightExampleWise(weights_, statisticIndex);
            }

            /**
             * @see `IStatisticsSubset::addToSubset`
             */
            void addToSubset(uint32 statisticIndex) override final {
                addExampleWiseStatisticToSubset(weights_, statisticView_, sumVector_, labelIndices_, statisticIndex);
            }

            /**
             * @see `IStatisticsSubset::calculateScores`
             */
            const IScoreVector& calculateScores() override final {
                return ruleEvaluationPtr_->calculateScores(sumVector_);
            }
    };

    /**
     * An abstract base class for all statistics that provide access to gradients and Hessians that are calculated
     * according to a differentiable loss function that is applied example-wise.
     *
     * @tparam StatisticVector          The type of the vectors that are used to store gradients and Hessians
     * @tparam StatisticView            The type of the view that provides access to the gradients and Hessians
     * @tparam RuleEvaluationFactory    The type of the factory that allows to create instances of the class that is
     *                                  used for calculating the predictions of rules, as well as corresponding quality
     *                                  scores
     * @tparam WeightVector             The type of the vector that provides access to the weights of individual
     *                                  statistics
     */
    template<typename StatisticVector, typename StatisticView, typename RuleEvaluationFactory, typename WeightVector>
    class AbstractExampleWiseImmutableWeightedStatistics : virtual public IImmutableWeightedStatistics {
        protected:

            /**
             * An abstract base class for all subsets of the gradients and Hessians that are stored by an instance of
             * the class `AbstractExampleWiseImmutableWeightedStatistics`.
             *
             * @tparam IndexVector The type of the vector that provides access to the indices of the labels that are
             *                     included in the subset
             */
            template<typename IndexVector>
            class AbstractWeightedStatisticsSubset
                : public ExampleWiseStatisticsSubset<StatisticVector, StatisticView, RuleEvaluationFactory,
                                                     WeightVector, IndexVector>,
                  virtual public IWeightedStatisticsSubset {
                private:

                    StatisticVector tmpVector_;

                    std::unique_ptr<StatisticVector> accumulatedSumVectorPtr_;

                protected:

                    /**
                     * A pointer to an object of template type `StatisticVector` that stores the total sum of all
                     * gradients and Hessians.
                     */
                    const StatisticVector* totalSumVector_;

                public:

                    /**
                     * @param statistics        A reference to an object of type
                     *                          `AbstractExampleWiseImmutableWeightedStatistics` that stores the
                     *                          gradients and Hessians
                     * @param totalSumVector    A reference to an object of template type `StatisticVector` that stores
                     *                          the total sums of gradients and Hessians
                     * @param labelIndices      A reference to an object of template type `IndexVector` that provides
                     *                          access to the indices of the labels that are included in the subset
                     */
                    AbstractWeightedStatisticsSubset(const AbstractExampleWiseImmutableWeightedStatistics& statistics,
                                                     const StatisticVector& totalSumVector,
                                                     const IndexVector& labelIndices)
                        : ExampleWiseStatisticsSubset<StatisticVector, StatisticView, RuleEvaluationFactory,
                                                      WeightVector, IndexVector>(statistics.statisticView_,
                                                                                 statistics.ruleEvaluationFactory_,
                                                                                 statistics.weights_, labelIndices),
                          tmpVector_(StatisticVector(labelIndices.getNumElements())), totalSumVector_(&totalSumVector) {
                    }

                    /**
                     * @see `IWeightedStatisticsSubset::resetSubset`
                     */
                    void resetSubset() override final {
                        if (!accumulatedSumVectorPtr_) {
                            // Create a vector for storing the accumulated sums of gradients and Hessians, if
                            // necessary...
                            accumulatedSumVectorPtr_ = std::make_unique<StatisticVector>(this->sumVector_);
                        } else {
                            // Add the sums of gradients and Hessians to the accumulated sums of gradients and
                            // Hessians...
                            accumulatedSumVectorPtr_->add(
                              this->sumVector_.gradients_cbegin(), this->sumVector_.gradients_cend(),
                              this->sumVector_.hessians_cbegin(), this->sumVector_.hessians_cend());
                        }

                        // Reset the sums of gradients and Hessians to zero...
                        this->sumVector_.clear();
                    }

                    /**
                     * @see `IWeightedStatisticsSubset::calculateScoresAccumulated`
                     */
                    const IScoreVector& calculateScoresAccumulated() override final {
                        return this->ruleEvaluationPtr_->calculateScores(*accumulatedSumVectorPtr_);
                    }

                    /**
                     * @see `IWeightedStatisticsSubset::calculateScoresUncovered`
                     */
                    const IScoreVector& calculateScoresUncovered() override final {
                        tmpVector_.difference(totalSumVector_->gradients_cbegin(), totalSumVector_->gradients_cend(),
                                              totalSumVector_->hessians_cbegin(), totalSumVector_->hessians_cend(),
                                              this->labelIndices_, this->sumVector_.gradients_cbegin(),
                                              this->sumVector_.gradients_cend(), this->sumVector_.hessians_cbegin(),
                                              this->sumVector_.hessians_cend());
                        return this->ruleEvaluationPtr_->calculateScores(tmpVector_);
                    }

                    /**
                     * @see `IWeightedStatisticsSubset::calculateScoresUncoveredAccumulated`
                     */
                    const IScoreVector& calculateScoresUncoveredAccumulated() override final {
                        tmpVector_.difference(
                          totalSumVector_->gradients_cbegin(), totalSumVector_->gradients_cend(),
                          totalSumVector_->hessians_cbegin(), totalSumVector_->hessians_cend(), this->labelIndices_,
                          accumulatedSumVectorPtr_->gradients_cbegin(), accumulatedSumVectorPtr_->gradients_cend(),
                          accumulatedSumVectorPtr_->hessians_cbegin(), accumulatedSumVectorPtr_->hessians_cend());
                        return this->ruleEvaluationPtr_->calculateScores(tmpVector_);
                    }
            };

        protected:

            /**
             * A reference to an object of template type `StatisticView` that stores the gradients and Hessians.
             */
            const StatisticView& statisticView_;

            /**
             * A reference to an object of template type `RuleEvaluationFactory` that is used to create instances of the
             * class that is used for calculating the predictions of rules, as well as their overall quality.
             */
            const RuleEvaluationFactory& ruleEvaluationFactory_;

            /**
             * A reference to an object of template type `WeightVector` that provides access to the weights of
             * individual statistics.
             */
            const WeightVector& weights_;

        public:

            /**
             * @param statisticView         A reference to an object of template type `StatisticView` that provides
             *                              access to the gradients and Hessians
             * @param ruleEvaluationFactory A reference to an object of template type `RuleEvaluationFactory` that
             *                              allows to create instances of the class that should be used for calculating
             *                              the predictions of rules, as well as their overall quality
             * @param weights               A reference to an object of template type `WeightVector` that provides
             *                              access to the weights of individual statistics
             */
            AbstractExampleWiseImmutableWeightedStatistics(const StatisticView& statisticView,
                                                           const RuleEvaluationFactory& ruleEvaluationFactory,
                                                           const WeightVector& weights)
                : statisticView_(statisticView), ruleEvaluationFactory_(ruleEvaluationFactory), weights_(weights) {}

            /**
             * @see `IImmutableWeightedStatistics::getNumStatistics`
             */
            uint32 getNumStatistics() const override final {
                return statisticView_.getNumRows();
            }

            /**
             * @see `IImmutableWeightedStatistics::getNumLabels`
             */
            uint32 getNumLabels() const override final {
                return statisticView_.getNumCols();
            }
    };

    /**
     * Provides access to gradients and Hessians that are calculated according to a differentiable loss function that is
     * applied example-wise and are organized as a histogram.
     *
     * @tparam StatisticVector          The type of the vectors that are used to store gradients and Hessians
     * @tparam StatisticView            The type of the view that provides access to the original gradients and Hessians
     * @tparam Histogram                The type of a histogram that stores aggregated gradients and Hessians
     * @tparam RuleEvaluationFactory    The type of the factory that allows to create instances of the class that is
     *                                  used for calculating the predictions of rules, as well as corresponding quality
     *                                  scores
     * @tparam BinIndexVector           The type of the vector that stores the indices of the bins, individual examples
     *                                  have been assigned to
     * @tparam WeightVector             The type of the vector that provides access to the weights of individual
     *                                  statistics
     */
    template<typename StatisticVector, typename StatisticView, typename Histogram, typename RuleEvaluationFactory,
             typename BinIndexVector, typename WeightVector>
    class ExampleWiseHistogram final
        : virtual public IHistogram,
          public AbstractExampleWiseImmutableWeightedStatistics<StatisticVector, Histogram, RuleEvaluationFactory,
                                                                BinWeightVector> {
        private:

            /**
             * Provides access to a subset of the gradients and Hessians that are stored by an instance of the class
             * `ExampleWiseHistogram`.
             *
             * @tparam IndexVector The type of the vector that provides access to the indices of the labels that are
             *                     included in the subset
             */
            template<typename IndexVector>
            class WeightedStatisticsSubset final
                : public AbstractExampleWiseImmutableWeightedStatistics<
                    StatisticVector, Histogram, RuleEvaluationFactory,
                    BinWeightVector>::template AbstractWeightedStatisticsSubset<IndexVector> {
                private:

                    const ExampleWiseHistogram& histogram_;

                    std::unique_ptr<StatisticVector> totalCoverableSumVectorPtr_;

                public:

                    /**
                     * @param histogram         A reference to an object of type `ExampleWiseHistogram` that stores the
                     *                          gradients and Hessians
                     * @param totalSumVector    A reference to an object of template type `StatisticVector` that stores
                     *                          the total sums of gradients and Hessians
                     * @param labelIndices      A reference to an object of template type `IndexVector` that provides
                     *                          access to the indices of the labels that are included in the subset
                     */
                    WeightedStatisticsSubset(const ExampleWiseHistogram& histogram,
                                             const StatisticVector& totalSumVector, const IndexVector& labelIndices)
                        : AbstractExampleWiseImmutableWeightedStatistics<
                          StatisticVector, Histogram, RuleEvaluationFactory,
                          BinWeightVector>::template AbstractWeightedStatisticsSubset<IndexVector>(histogram,
                                                                                                   totalSumVector,
                                                                                                   labelIndices),
                          histogram_(histogram) {}

                    /**
                     * @see `IWeightedStatisticsSubset::addToMissing`
                     */
                    void addToMissing(uint32 statisticIndex) override {
                        // Create a vector for storing the totals sums of gradients and Hessians, if necessary...
                        if (!totalCoverableSumVectorPtr_) {
                            totalCoverableSumVectorPtr_ = std::make_unique<StatisticVector>(*this->totalSumVector_);
                            this->totalSumVector_ = totalCoverableSumVectorPtr_.get();
                        }

                        // Subtract the gradients and Hessians of the example at the given index (weighted by the given
                        // weight) from the total sums of gradients and Hessians...
                        removeExampleWiseStatistic(histogram_.originalWeights_, histogram_.originalStatisticView_,
                                                   *totalCoverableSumVectorPtr_, statisticIndex);
                    }
            };

            const std::unique_ptr<Histogram> histogramPtr_;

            const std::unique_ptr<BinWeightVector> binWeightVectorPtr_;

            const BinIndexVector& binIndexVector_;

            const StatisticView& originalStatisticView_;

            const WeightVector& originalWeights_;

            const StatisticVector& totalSumVector_;

        public:

            /**
             * @param histogramPtr          An unique pointer to an object of template type `Histogram` that stores the
             *                              gradients and Hessians in the histogram
             * @param binWeightVectorPtr    An unique pointer to an object of type `BinWeightVector` that stores the
             *                              weights of individual bins
             * @param binIndexVector        A reference to an object of template type `BinIndexVector` that stores the
             *                              indices of the bins, individual examples have been assigned to
             * @param originalStatisticView A reference to an object of template type `StatisticView` that provides
             *                              access to the original gradients and Hessians, the histogram was created
             *                              from
             * @param originalWeights       A reference to an object of template type `WeightVector` that provides
             *                              access to the weights of the original statistics, the histogram was created
             *                              from
             * @param totalSumVector        A reference to an object of template type `StatisticVector` that stores the
             *                              total sums of gradients and Hessians
             * @param ruleEvaluationFactory A reference to an object of type `RuleEvaluationFactory` that allows to
             *                              create instances of the class that should be used for calculating the
             *                              predictions of rules, as well as their overall quality
             */
            ExampleWiseHistogram(std::unique_ptr<Histogram> histogramPtr,
                                 std::unique_ptr<BinWeightVector> binWeightVectorPtr,
                                 const BinIndexVector& binIndexVector, const StatisticView& originalStatisticView,
                                 const WeightVector& originalWeights, const StatisticVector& totalSumVector,
                                 const RuleEvaluationFactory& ruleEvaluationFactory)
                : AbstractExampleWiseImmutableWeightedStatistics<StatisticVector, Histogram, RuleEvaluationFactory,
                                                                 BinWeightVector>(*histogramPtr, ruleEvaluationFactory,
                                                                                  *binWeightVectorPtr),
                  histogramPtr_(std::move(histogramPtr)), binWeightVectorPtr_(std::move(binWeightVectorPtr)),
                  binIndexVector_(binIndexVector), originalStatisticView_(originalStatisticView),
                  originalWeights_(originalWeights), totalSumVector_(totalSumVector) {}

            /**
             * @see `IHistogram::clear`
             */
            void clear() override {
                histogramPtr_->clear();
                binWeightVectorPtr_->clear();
            }

            /**
             * @see `IHistogram::getBinWeight`
             */
            uint32 getBinWeight(uint32 binIndex) const override {
                return (*binWeightVectorPtr_)[binIndex];
            }

            /**
             * @see `IHistogram::addToBin`
             */
            void addToBin(uint32 statisticIndex) override {
                float64 weight = originalWeights_[statisticIndex];

                if (weight > 0) {
                    uint32 binIndex = binIndexVector_.getBinIndex(statisticIndex);

                    if (binIndex != IBinIndexVector::BIN_INDEX_SPARSE) {
                        binWeightVectorPtr_->increaseWeight(binIndex);
                        histogramPtr_->addToRow(binIndex, originalStatisticView_.gradients_cbegin(statisticIndex),
                                                originalStatisticView_.gradients_cend(statisticIndex),
                                                originalStatisticView_.hessians_cbegin(statisticIndex),
                                                originalStatisticView_.hessians_cend(statisticIndex), weight);
                    }
                }
            }

            /**
             * @see `IImmutableWeightedStatistics::createSubset`
             */
            std::unique_ptr<IWeightedStatisticsSubset> createSubset(
              const CompleteIndexVector& labelIndices) const override {
                return std::make_unique<WeightedStatisticsSubset<CompleteIndexVector>>(*this, totalSumVector_,
                                                                                       labelIndices);
            }

            /**
             * @see `IImmutableWeightedStatistics::createSubset`
             */
            std::unique_ptr<IWeightedStatisticsSubset> createSubset(
              const PartialIndexVector& labelIndices) const override {
                return std::make_unique<WeightedStatisticsSubset<PartialIndexVector>>(*this, totalSumVector_,
                                                                                      labelIndices);
            }
    };

    template<typename WeightVector, typename StatisticView, typename StatisticVector>
    static inline void addExampleWiseStatistic(const WeightVector& weights, const StatisticView& statisticView,
                                               StatisticVector& statisticVector, uint32 statisticIndex) {
        float64 weight = weights[statisticIndex];
        statisticVector.add(statisticView.gradients_cbegin(statisticIndex),
                            statisticView.gradients_cend(statisticIndex), statisticView.hessians_cbegin(statisticIndex),
                            statisticView.hessians_cend(statisticIndex), weight);
    }

    template<typename StatisticView, typename StatisticVector>
    static inline void addExampleWiseStatistic(const EqualWeightVector& weights, const StatisticView& statisticView,
                                               StatisticVector& statisticVector, uint32 statisticIndex) {
        statisticVector.add(statisticView.gradients_cbegin(statisticIndex),
                            statisticView.gradients_cend(statisticIndex), statisticView.hessians_cbegin(statisticIndex),
                            statisticView.hessians_cend(statisticIndex));
    }

    template<typename WeightVector, typename StatisticView, typename StatisticVector>
    static inline void removeExampleWiseStatistic(const WeightVector& weights, const StatisticView& statisticView,
                                                  StatisticVector& statisticVector, uint32 statisticIndex) {
        float64 weight = weights[statisticIndex];
        statisticVector.remove(
          statisticView.gradients_cbegin(statisticIndex), statisticView.gradients_cend(statisticIndex),
          statisticView.hessians_cbegin(statisticIndex), statisticView.hessians_cend(statisticIndex), weight);
    }

    template<typename StatisticView, typename StatisticVector>
    static inline void removeExampleWiseStatistic(const EqualWeightVector& weights, const StatisticView& statisticView,
                                                  StatisticVector& statisticVector, uint32 statisticIndex) {
        statisticVector.remove(
          statisticView.gradients_cbegin(statisticIndex), statisticView.gradients_cend(statisticIndex),
          statisticView.hessians_cbegin(statisticIndex), statisticView.hessians_cend(statisticIndex));
    }

    template<typename StatisticVector, typename StatisticView, typename Histogram, typename RuleEvaluationFactory,
             typename BinIndexVector, typename WeightVector>
    static inline std::unique_ptr<IHistogram> createExampleWiseHistogramInternally(
      const BinIndexVector& binIndexVector, const StatisticView& originalStatisticView,
      const WeightVector& originalWeights, const StatisticVector& totalSumVector,
      const RuleEvaluationFactory& ruleEvaluationFactory, uint32 numBins) {
        std::unique_ptr<Histogram> histogramPtr =
          std::make_unique<Histogram>(numBins, originalStatisticView.getNumCols());
        std::unique_ptr<BinWeightVector> binWeightVectorPtr = std::make_unique<BinWeightVector>(numBins);
        return std::make_unique<ExampleWiseHistogram<StatisticVector, StatisticView, Histogram, RuleEvaluationFactory,
                                                     BinIndexVector, WeightVector>>(
          std::move(histogramPtr), std::move(binWeightVectorPtr), binIndexVector, originalStatisticView,
          originalWeights, totalSumVector, ruleEvaluationFactory);
    }

    /**
     * Provides access to weighted gradients and Hessians that are calculated according to a differentiable loss
     * function that is applied example-wise and allows to update the gradients and Hessians after a new rule has been
     * learned.
     *
     * @tparam StatisticVector          The type of the vectors that are used to store gradients and Hessians
     * @tparam StatisticView            The type of the view that provides access to the gradients and Hessians
     * @tparam Histogram                The type of a histogram that stores aggregated gradients and Hessians
     * @tparam RuleEvaluationFactory    The type of the factory that allows to create instances of the class that is
     *                                  used for calculating the predictions of rules, as well as corresponding quality
     *                                  scores
     * @tparam WeightVector             The type of the vector that provides access to the weights of individual
     *                                  statistics
     */
    template<typename StatisticVector, typename StatisticView, typename Histogram, typename RuleEvaluationFactory,
             typename WeightVector>
    class ExampleWiseWeightedStatistics final
        : virtual public IWeightedStatistics,
          public AbstractExampleWiseImmutableWeightedStatistics<StatisticVector, StatisticView, RuleEvaluationFactory,
                                                                WeightVector> {
        private:

            /**
             * Provides access to a subset of the gradients and Hessians that are stored by an instance of the class
             * `ExampleWiseWeightedStatistics`.
             *
             * @tparam IndexVector The type of the vector that provides access to the indices of the labels that are
             *                     included in the subset
             */
            template<typename IndexVector>
            class WeightedStatisticsSubset final
                : public AbstractExampleWiseImmutableWeightedStatistics<
                    StatisticVector, StatisticView, RuleEvaluationFactory,
                    WeightVector>::template AbstractWeightedStatisticsSubset<IndexVector> {
                private:

                    std::unique_ptr<StatisticVector> totalCoverableSumVectorPtr_;

                public:

                    /**
                     * @param statistics        A reference to an object of type `ExampleWiseWeightedStatistics` that
                     *                          stores the gradients and Hessians
                     * @param totalSumVector    A reference to an object of template type `StatisticVector` that stores
                     *                          the total sums of gradients and Hessians
                     * @param labelIndices      A reference to an object of template type `IndexVector` that provides
                     *                          access to the indices of the labels that are included in the subset
                     */
                    WeightedStatisticsSubset(const ExampleWiseWeightedStatistics& statistics,
                                             const StatisticVector& totalSumVector, const IndexVector& labelIndices)
                        : AbstractExampleWiseImmutableWeightedStatistics<
                          StatisticVector, StatisticView, RuleEvaluationFactory,
                          WeightVector>::template AbstractWeightedStatisticsSubset<IndexVector>(statistics,
                                                                                                totalSumVector,
                                                                                                labelIndices) {}

                    /**
                     * @see `IWeightedStatisticsSubset::addToMissing`
                     */
                    void addToMissing(uint32 statisticIndex) override {
                        // Create a vector for storing the totals sums of gradients and Hessians, if necessary...
                        if (!totalCoverableSumVectorPtr_) {
                            totalCoverableSumVectorPtr_ = std::make_unique<StatisticVector>(*this->totalSumVector_);
                            this->totalSumVector_ = totalCoverableSumVectorPtr_.get();
                        }

                        // Subtract the gradients and Hessians of the example at the given index (weighted by the given
                        // weight) from the total sums of gradients and Hessians...
                        removeExampleWiseStatistic(this->weights_, this->statisticView_, *totalCoverableSumVectorPtr_,
                                                   statisticIndex);
                    }
            };

            const std::unique_ptr<StatisticVector> totalSumVectorPtr_;

        public:

            /**
             * @param statisticView         A reference to an object of template type `StatisticView` that provides
             *                              access to the gradients and Hessians
             * @param ruleEvaluationFactory A reference to an object of template type `RuleEvaluationFactory` that
             *                              allows to create instances of the class that should be used for calculating
             *                              the predictions of rules, as well as their overall quality
             * @param weights               A reference to an object of template type `WeightVector` that provides
             *                              access to the weights of individual statistics
             */
            ExampleWiseWeightedStatistics(const StatisticView& statisticView,
                                          const RuleEvaluationFactory& ruleEvaluationFactory,
                                          const WeightVector& weights)
                : AbstractExampleWiseImmutableWeightedStatistics<StatisticVector, StatisticView, RuleEvaluationFactory,
                                                                 WeightVector>(statisticView, ruleEvaluationFactory,
                                                                               weights),
                  totalSumVectorPtr_(std::make_unique<StatisticVector>(statisticView.getNumCols(), true)) {
                uint32 numStatistics = weights.getNumElements();

                for (uint32 i = 0; i < numStatistics; i++) {
                    addExampleWiseStatistic(weights, statisticView, *totalSumVectorPtr_, i);
                }
            }

            /**
             * @param statistics A reference to an object of type `ExampleWiseWeightedStatistics` to be copied
             */
            ExampleWiseWeightedStatistics(const ExampleWiseWeightedStatistics& statistics)
                : AbstractExampleWiseImmutableWeightedStatistics<StatisticVector, StatisticView, RuleEvaluationFactory,
                                                                 WeightVector>(
                  statistics.statisticView_, statistics.ruleEvaluationFactory_, statistics.weights_),
                  totalSumVectorPtr_(std::make_unique<StatisticVector>(*statistics.totalSumVectorPtr_)) {}

            /**
             * @see `IWeightedStatistics::copy`
             */
            std::unique_ptr<IWeightedStatistics> copy() const override {
                return std::make_unique<ExampleWiseWeightedStatistics<StatisticVector, StatisticView, Histogram,
                                                                      RuleEvaluationFactory, WeightVector>>(*this);
            }

            /**
             * @see `IWeightedStatistics::resetCoveredStatistics`
             */
            void resetCoveredStatistics() override {
                totalSumVectorPtr_->clear();
            }

            /**
             * @see `IWeightedStatistics::addCoveredStatistic`
             */
            void addCoveredStatistic(uint32 statisticIndex) override {
                addExampleWiseStatistic(this->weights_, this->statisticView_, *totalSumVectorPtr_, statisticIndex);
            }

            /**
             * @see `IWeightedStatistics::removeCoveredStatistic`
             */
            void removeCoveredStatistic(uint32 statisticIndex) override {
                removeExampleWiseStatistic(this->weights_, this->statisticView_, *totalSumVectorPtr_, statisticIndex);
            }

            /**
             * @see `IWeightedStatistics::createHistogram`
             */
            std::unique_ptr<IHistogram> createHistogram(const DenseBinIndexVector& binIndexVector,
                                                        uint32 numBins) const override {
                return createExampleWiseHistogramInternally<StatisticVector, StatisticView, Histogram,
                                                            RuleEvaluationFactory, DenseBinIndexVector, WeightVector>(
                  binIndexVector, this->statisticView_, this->weights_, *totalSumVectorPtr_,
                  this->ruleEvaluationFactory_, numBins);
            }

            /**
             * @see `IWeightedStatistics::createHistogram`
             */
            std::unique_ptr<IHistogram> createHistogram(const DokBinIndexVector& binIndexVector,
                                                        uint32 numBins) const override {
                return createExampleWiseHistogramInternally<StatisticVector, StatisticView, Histogram,
                                                            RuleEvaluationFactory, DokBinIndexVector, WeightVector>(
                  binIndexVector, this->statisticView_, this->weights_, *totalSumVectorPtr_,
                  this->ruleEvaluationFactory_, numBins);
            }

            /**
             * @see `IImmutableWeightedStatistics::createSubset`
             */
            std::unique_ptr<IWeightedStatisticsSubset> createSubset(
              const CompleteIndexVector& labelIndices) const override {
                return std::make_unique<WeightedStatisticsSubset<CompleteIndexVector>>(*this, *totalSumVectorPtr_,
                                                                                       labelIndices);
            }

            /**
             * @see `IImmutableWeightedStatistics::createSubset`
             */
            std::unique_ptr<IWeightedStatisticsSubset> createSubset(
              const PartialIndexVector& labelIndices) const override {
                return std::make_unique<WeightedStatisticsSubset<PartialIndexVector>>(*this, *totalSumVectorPtr_,
                                                                                      labelIndices);
            }
    };

    template<typename LabelMatrix, typename StatisticView, typename ScoreMatrix, typename LossFunction>
    static inline void updateExampleWiseStatisticsInternally(uint32 statisticIndex, const LabelMatrix& labelMatrix,
                                                             StatisticView& statisticView, ScoreMatrix& scoreMatrix,
                                                             const LossFunction& lossFunction) {
        lossFunction.updateExampleWiseStatistics(statisticIndex, labelMatrix, scoreMatrix, statisticView);
    }

    /**
     * An abstract base class for all statistics that provide access to gradients and Hessians that are calculated
     * according to a differentiable loss function that is applied example-wise.
     *
     * @tparam LabelMatrix                      The type of the matrix that provides access to the labels of the
     *                                          training examples
     * @tparam StatisticVector                  The type of the vectors that are used to store gradients and Hessians
     * @tparam StatisticView                    The type of the view that provides access to the gradients and Hessians
     * @tparam Histogram                        The type of a histogram that stores aggregated gradients and Hessians
     * @tparam ScoreMatrix                      The type of the matrices that are used to store predicted scores
     * @tparam LossFunction                     The type of the loss function that is used to calculate gradients and
     *                                          Hessians
     * @tparam EvaluationMeasure                The type of the evaluation measure that is used to assess the quality of
     *                                          predictions for a specific statistic
     * @tparam ExampleWiseRuleEvaluationFactory The type of the factory that allows to create instances of the class
     *                                          that is used for calculating the example-wise predictions of rules, as
     *                                          well as their overall quality
     * @tparam LabelWiseRuleEvaluationFactory   The type of the factory that allows to create instances of the class
     *                                          that is used for calculating the label-wise predictions of rules, as
     *                                          well as their overall quality
     */
    template<typename LabelMatrix, typename StatisticVector, typename StatisticView, typename Histogram,
             typename ScoreMatrix, typename LossFunction, typename EvaluationMeasure,
             typename ExampleWiseRuleEvaluationFactory, typename LabelWiseRuleEvaluationFactory>
    class AbstractExampleWiseStatistics
        : virtual public IExampleWiseStatistics<ExampleWiseRuleEvaluationFactory, LabelWiseRuleEvaluationFactory> {
        private:

            const ExampleWiseRuleEvaluationFactory* ruleEvaluationFactory_;

        protected:

            /**
             * An unique pointer to the loss function that should be used for calculating gradients and Hessians.
             */
            std::unique_ptr<LossFunction> lossPtr_;

            /**
             * An unique pointer to the evaluation measure that should be used to assess the quality of predictions for
             * a specific statistic.
             */
            std::unique_ptr<EvaluationMeasure> evaluationMeasurePtr_;

            /**
             * The label matrix that provides access to the labels of the training examples.
             */
            const LabelMatrix& labelMatrix_;

            /**
             * An unique pointer to an object of template type `StatisticView` that stores the gradients and Hessians.
             */
            const std::unique_ptr<StatisticView> statisticViewPtr_;

            /**
             * The score matrix that stores the currently predicted scores.
             */
            std::unique_ptr<ScoreMatrix> scoreMatrixPtr_;

        public:

            /**
             * @param lossPtr               An unique pointer to an object of template type `LossFunction` that
             *                              implements the loss function that should be used for calculating gradients
             *                              and Hessians
             * @param evaluationMeasurePtr  An unique pointer to an object of template type `EvaluationMeasure` that
             *                              implements the evaluation measure that should be used to assess the quality
             *                              of predictions for a specific statistic
             * @param ruleEvaluationFactory A reference to an object of template type `ExampleWiseRuleEvaluationFactory`
             *                              that allows to create instances of the class that should be used for
             *                              calculating the predictions of rules, as well as corresponding quality
             *                              scores
             * @param labelMatrix           A reference to an object of template type `LabelMatrix` that provides access
             *                              to the labels of the training examples
             * @param statisticViewPtr      An unique pointer to an object of template type `StatisticView` that
             *                              provides access to the gradients and Hessians
             * @param scoreMatrixPtr        An unique pointer to an object of template type `ScoreMatrix` that stores
             *                              the currently predicted scores
             */
            AbstractExampleWiseStatistics(std::unique_ptr<LossFunction> lossPtr,
                                          std::unique_ptr<EvaluationMeasure> evaluationMeasurePtr,
                                          const ExampleWiseRuleEvaluationFactory& ruleEvaluationFactory,
                                          const LabelMatrix& labelMatrix,
                                          std::unique_ptr<StatisticView> statisticViewPtr,
                                          std::unique_ptr<ScoreMatrix> scoreMatrixPtr)
                : ruleEvaluationFactory_(&ruleEvaluationFactory), lossPtr_(std::move(lossPtr)),
                  evaluationMeasurePtr_(std::move(evaluationMeasurePtr)), labelMatrix_(labelMatrix),
                  statisticViewPtr_(std::move(statisticViewPtr)), scoreMatrixPtr_(std::move(scoreMatrixPtr)) {}

            /**
             * @see `IExampleWiseStatistics::setRuleEvaluationFactory`
             */
            void setRuleEvaluationFactory(
              const ExampleWiseRuleEvaluationFactory& ruleEvaluationFactory) override final {
                this->ruleEvaluationFactory_ = &ruleEvaluationFactory;
            }

            /**
             * @see `IStatistics::getNumStatistics`
             */
            uint32 getNumStatistics() const override final {
                return statisticViewPtr_->getNumRows();
            }

            /**
             * @see `IStatistics::getNumLabels`
             */
            uint32 getNumLabels() const override final {
                return statisticViewPtr_->getNumCols();
            }

            /**
             * @see `IStatistics::applyPrediction`
             */
            void applyPrediction(uint32 statisticIndex, const CompletePrediction& prediction) override final {
                applyPredictionInternally(statisticIndex, prediction, *scoreMatrixPtr_);
                updateExampleWiseStatisticsInternally(statisticIndex, labelMatrix_, *this->statisticViewPtr_,
                                                      *scoreMatrixPtr_, *lossPtr_);
            }

            /**
             * @see `IStatistics::applyPrediction`
             */
            void applyPrediction(uint32 statisticIndex, const PartialPrediction& prediction) override final {
                applyPredictionInternally(statisticIndex, prediction, *scoreMatrixPtr_);
                updateExampleWiseStatisticsInternally(statisticIndex, labelMatrix_, *this->statisticViewPtr_,
                                                      *scoreMatrixPtr_, *lossPtr_);
            }

            /**
             * @see `IStatistics::revertPrediction`
             */
            void revertPrediction(uint32 statisticIndex, const CompletePrediction& prediction) override final {
                revertPredictionInternally(statisticIndex, prediction, *scoreMatrixPtr_);
                updateExampleWiseStatisticsInternally(statisticIndex, labelMatrix_, *this->statisticViewPtr_,
                                                      *scoreMatrixPtr_, *lossPtr_);
            }

            /**
             * @see `IStatistics::revertPrediction`
             */
            void revertPrediction(uint32 statisticIndex, const PartialPrediction& prediction) override final {
                revertPredictionInternally(statisticIndex, prediction, *scoreMatrixPtr_);
                updateExampleWiseStatisticsInternally(statisticIndex, labelMatrix_, *this->statisticViewPtr_,
                                                      *scoreMatrixPtr_, *lossPtr_);
            }

            /**
             * @see `IStatistics::evaluatePrediction`
             */
            float64 evaluatePrediction(uint32 statisticIndex) const override final {
                return evaluationMeasurePtr_->evaluate(statisticIndex, labelMatrix_, *scoreMatrixPtr_);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const CompleteIndexVector& labelIndices,
                                                            const EqualWeightVector& weights) const override final {
                return std::make_unique<
                  ExampleWiseStatisticsSubset<StatisticVector, StatisticView, ExampleWiseRuleEvaluationFactory,
                                              EqualWeightVector, CompleteIndexVector>>(
                  *statisticViewPtr_, *ruleEvaluationFactory_, weights, labelIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const PartialIndexVector& labelIndices,
                                                            const EqualWeightVector& weights) const override final {
                return std::make_unique<
                  ExampleWiseStatisticsSubset<StatisticVector, StatisticView, ExampleWiseRuleEvaluationFactory,
                                              EqualWeightVector, PartialIndexVector>>(
                  *statisticViewPtr_, *ruleEvaluationFactory_, weights, labelIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const CompleteIndexVector& labelIndices,
                                                            const BitWeightVector& weights) const override final {
                return std::make_unique<
                  ExampleWiseStatisticsSubset<StatisticVector, StatisticView, ExampleWiseRuleEvaluationFactory,
                                              BitWeightVector, CompleteIndexVector>>(
                  *statisticViewPtr_, *ruleEvaluationFactory_, weights, labelIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const PartialIndexVector& labelIndices,
                                                            const BitWeightVector& weights) const override final {
                return std::make_unique<
                  ExampleWiseStatisticsSubset<StatisticVector, StatisticView, ExampleWiseRuleEvaluationFactory,
                                              BitWeightVector, PartialIndexVector>>(
                  *statisticViewPtr_, *ruleEvaluationFactory_, weights, labelIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const CompleteIndexVector& labelIndices, const DenseWeightVector<uint32>& weights) const override final {
                return std::make_unique<
                  ExampleWiseStatisticsSubset<StatisticVector, StatisticView, ExampleWiseRuleEvaluationFactory,
                                              DenseWeightVector<uint32>, CompleteIndexVector>>(
                  *statisticViewPtr_, *ruleEvaluationFactory_, weights, labelIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const PartialIndexVector& labelIndices, const DenseWeightVector<uint32>& weights) const override final {
                return std::make_unique<
                  ExampleWiseStatisticsSubset<StatisticVector, StatisticView, ExampleWiseRuleEvaluationFactory,
                                              DenseWeightVector<uint32>, PartialIndexVector>>(
                  *statisticViewPtr_, *ruleEvaluationFactory_, weights, labelIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const CompleteIndexVector& labelIndices,
              const OutOfSampleWeightVector<EqualWeightVector>& weights) const override final {
                return std::make_unique<
                  ExampleWiseStatisticsSubset<StatisticVector, StatisticView, ExampleWiseRuleEvaluationFactory,
                                              OutOfSampleWeightVector<EqualWeightVector>, CompleteIndexVector>>(
                  *statisticViewPtr_, *ruleEvaluationFactory_, weights, labelIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const PartialIndexVector& labelIndices,
              const OutOfSampleWeightVector<EqualWeightVector>& weights) const override final {
                return std::make_unique<
                  ExampleWiseStatisticsSubset<StatisticVector, StatisticView, ExampleWiseRuleEvaluationFactory,
                                              OutOfSampleWeightVector<EqualWeightVector>, PartialIndexVector>>(
                  *statisticViewPtr_, *ruleEvaluationFactory_, weights, labelIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const CompleteIndexVector& labelIndices,
              const OutOfSampleWeightVector<BitWeightVector>& weights) const override final {
                return std::make_unique<
                  ExampleWiseStatisticsSubset<StatisticVector, StatisticView, ExampleWiseRuleEvaluationFactory,
                                              OutOfSampleWeightVector<BitWeightVector>, CompleteIndexVector>>(
                  *statisticViewPtr_, *ruleEvaluationFactory_, weights, labelIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const PartialIndexVector& labelIndices,
              const OutOfSampleWeightVector<BitWeightVector>& weights) const override final {
                return std::make_unique<
                  ExampleWiseStatisticsSubset<StatisticVector, StatisticView, ExampleWiseRuleEvaluationFactory,
                                              OutOfSampleWeightVector<BitWeightVector>, PartialIndexVector>>(
                  *statisticViewPtr_, *ruleEvaluationFactory_, weights, labelIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const CompleteIndexVector& labelIndices,
              const OutOfSampleWeightVector<DenseWeightVector<uint32>>& weights) const override final {
                return std::make_unique<
                  ExampleWiseStatisticsSubset<StatisticVector, StatisticView, ExampleWiseRuleEvaluationFactory,
                                              OutOfSampleWeightVector<DenseWeightVector<uint32>>, CompleteIndexVector>>(
                  *statisticViewPtr_, *ruleEvaluationFactory_, weights, labelIndices);
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const PartialIndexVector& labelIndices,
              const OutOfSampleWeightVector<DenseWeightVector<uint32>>& weights) const override final {
                return std::make_unique<
                  ExampleWiseStatisticsSubset<StatisticVector, StatisticView, ExampleWiseRuleEvaluationFactory,
                                              OutOfSampleWeightVector<DenseWeightVector<uint32>>, PartialIndexVector>>(
                  *statisticViewPtr_, *ruleEvaluationFactory_, weights, labelIndices);
            }

            /**
             * @see `IStatistics::createWeightedStatistics`
             */
            std::unique_ptr<IWeightedStatistics> createWeightedStatistics(
              const EqualWeightVector& weights) const override final {
                return std::make_unique<ExampleWiseWeightedStatistics<
                  StatisticVector, StatisticView, Histogram, ExampleWiseRuleEvaluationFactory, EqualWeightVector>>(
                  *statisticViewPtr_, *ruleEvaluationFactory_, weights);
            }

            /**
             * @see `IStatistics::createWeightedStatistics`
             */
            std::unique_ptr<IWeightedStatistics> createWeightedStatistics(
              const BitWeightVector& weights) const override final {
                return std::make_unique<ExampleWiseWeightedStatistics<
                  StatisticVector, StatisticView, Histogram, ExampleWiseRuleEvaluationFactory, BitWeightVector>>(
                  *statisticViewPtr_, *ruleEvaluationFactory_, weights);
            }

            /**
             * @see `IStatistics::createWeightedStatistics`
             */
            std::unique_ptr<IWeightedStatistics> createWeightedStatistics(
              const DenseWeightVector<uint32>& weights) const override final {
                return std::make_unique<
                  ExampleWiseWeightedStatistics<StatisticVector, StatisticView, Histogram,
                                                ExampleWiseRuleEvaluationFactory, DenseWeightVector<uint32>>>(
                  *statisticViewPtr_, *ruleEvaluationFactory_, weights);
            }
    };

}
