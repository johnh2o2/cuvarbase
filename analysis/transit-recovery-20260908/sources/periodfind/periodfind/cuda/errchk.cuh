// Copyright 2021 California Institute of Technology. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
// Author: Ethan Jaszewski

#ifndef __PF_ERRCHK_H__
#define __PF_ERRCHK_H__

#include <iostream>

// Macro modified from one found on StackOverflow.
// See: https://stackoverflow.com/questions/14038589/

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code,
                      const char* file,
                      int line,
                      bool abort = true) {
    if (code != cudaSuccess) {
        std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file
                  << " " << line << "\n";
        if (abort)
            exit(code);
    }
}

#endif
