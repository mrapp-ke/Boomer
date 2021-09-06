/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/model/body.hpp"


/**
 * An empty body that does not contain any conditions and therefore covers any examples.
 */
class EmptyBody final : public IBody {

    public:

        bool covers(CContiguousFeatureMatrix::const_iterator begin,
                    CContiguousFeatureMatrix::const_iterator end) const override;

        bool covers(CsrFeatureMatrix::index_const_iterator indicesBegin,
                    CsrFeatureMatrix::index_const_iterator indicesEnd,
                    CsrFeatureMatrix::value_const_iterator valuesBegin,
                    CsrFeatureMatrix::value_const_iterator valuesEnd, float32* tmpArray1, uint32* tmpArray2,
                    uint32 n) const override;

        void visit(EmptyBodyVisitor emptyBodyVisitor, ConjunctiveBodyVisitor conjunctiveBodyVisitor) const override;

};
