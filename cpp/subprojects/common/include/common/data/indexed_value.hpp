/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/types.hpp"


/**
 * A tuple that consists of an index and a value.
 */
template<class T>
struct IndexedValue {

    /**
     * The index.
     */
    uint32 index;

    /**
     * The value.
     */
    T value;

};
