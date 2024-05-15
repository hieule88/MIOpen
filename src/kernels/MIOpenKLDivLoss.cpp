/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"
#include "tensor_view.hpp"

#ifndef INPUT_TYPE
#define INPUT_TYPE float
#endif

#ifndef OUTPUT_TYPE
#define OUTPUT_TYPE float
#endif

#ifndef D_TYPE
#define D_TYPE float
#endif

#ifndef REDUCE_SIZE
#define REDUCE_SIZE 256
#endif

template <typename TI, typename TO>
__device__ void kldivLossUnreducedForward5d(const TI* __restrict__ input,
                                          const TI* __restrict__ target,
                                          TO* __restrict__ output,
                                          bool log_target,
                                          tensor_view_5d_t input_tv,
                                          tensor_view_5d_t target_tv,
                                          tensor_view_5d_t output_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    size_t n[5];
    GET_NCDHW(n[0], n[1], n[2], n[3], n[4], gid, output_tv);

    if(n[0] >= output_tv.size[0])
        return;

    size_t Iidx = TV5D_IDX(input_tv, n[0], n[1], n[2], n[3], n[4]);
    size_t Tidx = TV5D_IDX(target_tv, n[0], n[1], n[2], n[3], n[4]);
    size_t Oidx = TV5D_IDX(output_tv, n[0], n[1], n[2], n[3], n[4]);

    FLOAT_ACCUM input_value = CVT_FLOAT2ACCUM(input[Iidx]);
    FLOAT_ACCUM target_value = CVT_FLOAT2ACCUM(target[Tidx]);
    FLOAT_ACCUM forward_output;

    if (log_target) {
        forward_output = exp(target_value)
                                    * (target_value - input_value);
        output[Oidx] = isnan(forward_output) ? CVT_FP32_2FLOAT(0.0f) : CVT_ACCUM2FLOAT(forward_output);
    } else {
        forward_output = target_value 
                                    * (log(target_value) - input_value);
        output[Oidx] = isnan(forward_output) ? CVT_FP32_2FLOAT(0.0f) : CVT_ACCUM2FLOAT(forward_output);
    }
}

extern "C" __global__ void KLDivLossUnreducedForward5d(const INPUT_TYPE* __restrict__ input,
                                                     const INPUT_TYPE* __restrict__ target,
                                                     OUTPUT_TYPE* __restrict__ output,
                                                     bool log_target,
                                                     tensor_view_5d_t input_tv,
                                                     tensor_view_5d_t target_tv,
                                                     tensor_view_5d_t output_tv)
{
    kldivLossUnreducedForward5d<INPUT_TYPE, OUTPUT_TYPE>(
        input, target, output, log_target, input_tv, target_tv, output_tv);
}
