#include "boosting/data/statistic_view_label_wise_sparse.hpp"

namespace boosting {

    SparseLabelWiseStatisticConstView::SparseLabelWiseStatisticConstView(uint32 numCols,
                                                                         SparseSetMatrix<Tuple<float64>>* statistics)
        : numCols_(numCols), statistics_(statistics) {}

    SparseLabelWiseStatisticConstView::const_iterator SparseLabelWiseStatisticConstView::cbegin(uint32 row) const {
        return statistics_->cbegin(row);
    }

    SparseLabelWiseStatisticConstView::const_iterator SparseLabelWiseStatisticConstView::cend(uint32 row) const {
        return statistics_->cend(row);
    }

    SparseLabelWiseStatisticConstView::const_row SparseLabelWiseStatisticConstView::operator[](uint32 row) const {
        return ((const SparseSetMatrix<Tuple<float64>>&) *statistics_)[row];
    }

    uint32 SparseLabelWiseStatisticConstView::getNumRows() const {
        return statistics_->getNumRows();
    }

    uint32 SparseLabelWiseStatisticConstView::getNumCols() const {
        return numCols_;
    }

    SparseLabelWiseStatisticView::SparseLabelWiseStatisticView(uint32 numCols,
                                                               SparseSetMatrix<Tuple<float64>>* statistics)
        : SparseLabelWiseStatisticConstView(numCols, statistics) {}

    SparseLabelWiseStatisticView::row SparseLabelWiseStatisticView::operator[](uint32 row) {
        return (*statistics_)[row];
    }

    void SparseLabelWiseStatisticView::clear() {
        statistics_->clear();
    }

}
