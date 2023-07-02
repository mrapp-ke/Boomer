#include "boosting/data/histogram_view_label_wise_sparse.hpp"

#include "common/data/arrays.hpp"
#include "statistic_vector_label_wise_sparse_common.hpp"

namespace boosting {

    SparseLabelWiseHistogramConstView::SparseLabelWiseHistogramConstView(uint32 numRows, uint32 numCols,
                                                                         Triple<float64>* statistics, float64* weights)
        : numRows_(numRows), numCols_(numCols), statistics_(statistics), weights_(weights) {}

    SparseLabelWiseHistogramConstView::const_iterator SparseLabelWiseHistogramConstView::cbegin(uint32 row) const {
        return &statistics_[row * numCols_];
    }

    SparseLabelWiseHistogramConstView::const_iterator SparseLabelWiseHistogramConstView::cend(uint32 row) const {
        return &statistics_[(row + 1) * numCols_];
    }

    SparseLabelWiseHistogramConstView::weight_const_iterator SparseLabelWiseHistogramConstView::weights_cbegin() const {
        return weights_;
    }

    SparseLabelWiseHistogramConstView::weight_const_iterator SparseLabelWiseHistogramConstView::weights_cend() const {
        return &weights_[numRows_];
    }

    uint32 SparseLabelWiseHistogramConstView::getNumRows() const {
        return numRows_;
    }

    uint32 SparseLabelWiseHistogramConstView::getNumCols() const {
        return numCols_;
    }

    SparseLabelWiseHistogramView::SparseLabelWiseHistogramView(uint32 numRows, uint32 numCols,
                                                               Triple<float64>* statistics, float64* weights)
        : SparseLabelWiseHistogramConstView(numRows, numCols, statistics, weights) {}

    void SparseLabelWiseHistogramView::clear() {
        setArrayToZeros(weights_, numRows_);
        setArrayToZeros(statistics_, numRows_ * numCols_);
    }

    void SparseLabelWiseHistogramView::addToRow(uint32 row, SparseLabelWiseStatisticConstView::const_iterator begin,
                                                SparseLabelWiseStatisticConstView::const_iterator end, float64 weight) {
        if (weight != 0) {
            weights_[row] += weight;
            addToSparseLabelWiseStatisticVector(&statistics_[row * numCols_], begin, end, weight);
        }
    }

}
