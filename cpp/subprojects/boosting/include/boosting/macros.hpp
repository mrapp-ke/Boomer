/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#ifdef _WIN32
    #ifdef MLRLBOOSTING_EXPORTS
        #define MLRLBOOSTING_API __declspec(dllexport)
    #else
        #define MLRLBOOSTING_API __declspec(dllimport)
    #endif
#else
    #define MLRLBOOSTING_API
#endif
