/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/list_of_lists.hpp"

/**
 * A two-dimensional matrix that provides row-wise access to binary values that are stored in the list of lists (LIL)
 * format.
 */
typedef ListOfLists<uint32> BinaryLilMatrix;
