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

template <typename TI, typename TO>
__device__ void typeCast(const TI* __restrict__ input,
                         TO* __restrict__ output,
                         uint64_t bits_to_truncate,
                         uint64_t numel,
                         tensor_view_t<5> input_tv,
                         tensor_view_t<5> output_tv)
{
    uint64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= numel)
    {
        return;
    }

    tensor_layout_t<5> layout(input_tv, gid);
    TI val = input[input_tv.get_tensor_view_idx(layout)];
    if(bits_to_truncate)
    {
        val = truncate_func(gid, val, bits_to_truncate);
    }
    output[output_tv.get_tensor_view_idx(layout)] = cast_func(val);
}

extern "C" __global__ void TypeCast(const I_TYPE* __restrict__ input,
                                    O_TYPE* __restrict__ output,
                                    uint64_t bits_to_truncate,
                                    uint64_t numel,
                                    tensor_view_t<5> input_tv,
                                    tensor_view_t<5> output_tv)
{
    typeCast<I_TYPE, O_TYPE>(input, output, bits_to_truncate, numel, input_tv, output_tv);
}

template <typename TI, typename TO>
__device__ void typeCastContiguous(const TI* __restrict__ input,
                                   TO* __restrict__ output,
                                   uint64_t bits_to_truncate,
                                   uint64_t numel)
{
    uint64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= numel)
    {
        return;
    }

    TI val = input[gid];
    if(bits_to_truncate)
    {
        val = truncate_func(gid, val, bits_to_truncate);
    }
    output[gid] = cast_func(val);
}

extern "C" __global__ void TypeCastContiguous(const I_TYPE* __restrict__ input,
                                              O_TYPE* __restrict__ output,
                                              uint64_t bits_to_truncate,
                                              uint64_t numel)
{
    typeCastContiguous<I_TYPE, O_TYPE>(input, output, bits_to_truncate, numel);
}

inline DTYPE truncate_func(size_t gid, DTYPE val, int nbits)
{
#if IS_DTYPE_FLOAT
    return as_float((as_uint(val) & ((uint)0xFFFFFFFF << nbits)));
#elif IS_DTYPE_DOUBLE
    return as_double((as_ulong(val) & ((ulong)0xFFFFFFFFFFFFFFFF << nbits)));
#else
    return val;
#endif
}
