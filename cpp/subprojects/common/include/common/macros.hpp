/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#ifdef _WIN32
    #ifdef MLRLCOMMON_EXPORTS
        #define MLRLCOMMON_API __declspec(dllexport)
    #else
        #define MLRLCOMMON_API __declspec(dllimport)
    #endif
#else
    #define MLRLCOMMON_API
#endif
