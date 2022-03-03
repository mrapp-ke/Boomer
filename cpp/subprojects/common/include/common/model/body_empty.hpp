/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/model/body.hpp"


/**
 * An empty body that does not contain any conditions and therefore covers any examples.
 */
class MLRLCOMMON_API EmptyBody final : public IBody {

    public:

        bool covers(CContiguousConstView<const float32>::value_const_iterator begin,
                    CContiguousConstView<const float32>::value_const_iterator end) const override;

        bool covers(CsrConstView<const float32>::index_const_iterator indicesBegin,
                    CsrConstView<const float32>::index_const_iterator indicesEnd,
                    CsrConstView<const float32>::value_const_iterator valuesBegin,
                    CsrConstView<const float32>::value_const_iterator valuesEnd, float32* tmpArray1, uint32* tmpArray2,
                    uint32 n) const override;

        void visit(EmptyBodyVisitor emptyBodyVisitor, ConjunctiveBodyVisitor conjunctiveBodyVisitor) const override;

};
