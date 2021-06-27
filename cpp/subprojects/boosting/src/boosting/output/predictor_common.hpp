#pragma once
#include "common/model/head_full.hpp"
#include "common/model/head_partial.hpp"


namespace boosting {

    static inline void applyFullHead(const FullHead& head, CContiguousView<float64>::iterator iterator) {
        FullHead::score_const_iterator scoreIterator = head.scores_cbegin();
        uint32 numElements = head.getNumElements();

        for (uint32 i = 0; i < numElements; i++) {
            iterator[i] += scoreIterator[i];
        }
    }

    static inline void applyPartialHead(const PartialHead& head, CContiguousView<float64>::iterator iterator) {
        PartialHead::score_const_iterator scoreIterator = head.scores_cbegin();
        PartialHead::index_const_iterator indexIterator = head.indices_cbegin();
        uint32 numElements = head.getNumElements();

        for (uint32 i = 0; i < numElements; i++) {
            uint32 index = indexIterator[i];
            iterator[index] += scoreIterator[i];
        }
    }

    static inline void applyHead(const IHead& head, CContiguousView<float64>::iterator scoreIterator) {
        auto fullHeadVisitor = [=](const FullHead& head) {
            applyFullHead(head, scoreIterator);
        };
        auto partialHeadVisitor = [=](const PartialHead& head) {
            applyPartialHead(head, scoreIterator);
        };
        head.visit(fullHeadVisitor, partialHeadVisitor);
    }

    static inline void applyRule(const Rule& rule, CContiguousFeatureMatrix::const_iterator featureValuesBegin,
                                 CContiguousFeatureMatrix::const_iterator featureValuesEnd,
                                 CContiguousView<float64>::iterator scoreIterator) {
        const IBody& body = rule.getBody();

        if (body.covers(featureValuesBegin, featureValuesEnd)) {
            const IHead& head = rule.getHead();
            applyHead(head, scoreIterator);
        }
    }

    static inline void applyRuleCsr(const Rule& rule, CsrFeatureMatrix::index_const_iterator featureIndicesBegin,
                                    CsrFeatureMatrix::index_const_iterator featureIndicesEnd,
                                    CsrFeatureMatrix::value_const_iterator featureValuesBegin,
                                    CsrFeatureMatrix::value_const_iterator featureValuesEnd,
                                    CContiguousView<float64>::iterator scoreIterator, float32* tmpArray1,
                                    uint32* tmpArray2, uint32 n) {
        const IBody& body = rule.getBody();

        if (body.covers(featureIndicesBegin, featureIndicesEnd, featureValuesBegin, featureValuesEnd, tmpArray1,
                        tmpArray2, n)) {
            const IHead& head = rule.getHead();
            applyHead(head, scoreIterator);
        }
    }

}
