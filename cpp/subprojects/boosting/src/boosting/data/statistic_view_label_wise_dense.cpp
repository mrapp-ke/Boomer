#include "boosting/data/statistic_view_label_wise_dense.hpp"

#include "boosting/data/arrays.hpp"
#include "common/data/arrays.hpp"

namespace boosting {

    DenseLabelWiseStatisticConstView::DenseLabelWiseStatisticConstView(uint32 numRows, uint32 numCols,
                                                                       Tuple<float64>* statistics)
        : numRows_(numRows), numCols_(numCols), statistics_(statistics) {}

    DenseLabelWiseStatisticConstView::const_iterator DenseLabelWiseStatisticConstView::cbegin(uint32 row) const {
        return &statistics_[row * numCols_];
    }

    DenseLabelWiseStatisticConstView::const_iterator DenseLabelWiseStatisticConstView::cend(uint32 row) const {
        return &statistics_[(row + 1) * numCols_];
    }

    uint32 DenseLabelWiseStatisticConstView::getNumRows() const {
        return numRows_;
    }

    uint32 DenseLabelWiseStatisticConstView::getNumCols() const {
        return numCols_;
    }

    DenseLabelWiseStatisticView::DenseLabelWiseStatisticView(uint32 numRows, uint32 numCols, Tuple<float64>* statistics)
        : DenseLabelWiseStatisticConstView(numRows, numCols, statistics) {}

    DenseLabelWiseStatisticView::iterator DenseLabelWiseStatisticView::begin(uint32 row) {
        return &statistics_[row * numCols_];
    }

    DenseLabelWiseStatisticView::iterator DenseLabelWiseStatisticView::end(uint32 row) {
        return &statistics_[(row + 1) * numCols_];
    }

    void DenseLabelWiseStatisticView::clear() {
        setArrayToZeros(statistics_, numRows_ * numCols_);
    }

    void DenseLabelWiseStatisticView::addToRow(uint32 row, const_iterator begin, const_iterator end, float64 weight) {
        uint32 offset = row * numCols_;
        addToArray(&statistics_[offset], begin, numCols_, weight);
    }

}
