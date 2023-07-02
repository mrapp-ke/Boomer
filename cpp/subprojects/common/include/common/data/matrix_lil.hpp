/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/indexed_value.hpp"
#include "common/data/list_of_lists.hpp"

/**
 * A two-dimensional matrix that provides row-wise access to data that is stored in the list of lists (LIL) format.
 *
 * @tparam T The type of the data that is stored by the matrix
 */
template<typename T>
using LilMatrix = ListOfLists<IndexedValue<T>>;
