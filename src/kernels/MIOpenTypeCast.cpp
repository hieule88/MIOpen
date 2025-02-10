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
#if MIOPEN_USE_FP32 == 1
    return as_float((as_uint(val) & (static_cast<uint>(0xFFFFFFFF) << nbits)));
#elif MIOPEN_USE_FP64 == 1
    return as_double((as_ulong(val) & ((ulong)0xFFFFFFFFFFFFFFFF << nbits)));
#else
    return val;
#endif
}

template <typename T>
__device__ bool convert_bool(T x)
{
    return static_cast<bool>(x);
}

template <typename T>
__device__ ushort convert_bfloat16(T x)
{
#if MIOPEN_USE_FP32 == 1
    return (as_uint(x) + ((as_uint(x) >> 16) & 1) + (static_cast<uint>(0x7FFF))) >> 16;
#else
    return (as_uint(convert_float(x)) + ((as_uint(convert_float(x)) >> 16) & 1) +
            (static_cast<uint>(0x7FFF))) >>
           16;
#endif
}

template <typename T>
__device__ half2 convert_complex32(T x)
{
    return (half2)(convert_half(x), (half)0);
}

template <typename T>
__device__ float2 convert_complex64(T x)
{
    return (float2)(convert_float(x), (float)0);
}

template <typename T>
__device__ double2 convert_complex128(T x)
{
    return (double2)(convert_double(x), (double)0);
}

#define TYPECAST(dest_type, dest_type_ptr, cast_func)                                             \
    extern "C" __global__ void Typecast_##dest_type(const I_TYPE* __restrict__ input,             \
                                                    dest_type_ptr __restrict__ output,            \
                                                    uint64_t numel,                               \
                                                    uint64_t bits_to_truncate)                    \
    {                                                                                             \
        typeCast<I_TYPE, dest_type>(input, output, bits_to_truncate, numel, input_tv, output_tv); \
    }

#define TYPECASTCONTIGUOUS(dest_type, dest_type_ptr, cast_func)                                  \
    extern "C" __global__ void TypecastContiguous_##dest_type(const I_TYPE* __restrict__ input,  \
                                                              dest_type_ptr __restrict__ output, \
                                                              uint64_t numel,                    \
                                                              uint64_t bits_to_truncate)         \
    {                                                                                            \
        typeCastContiguous<I_TYPE, dest_type>(input, output, bits_to_truncate, numel);           \
    }

TYPECAST(bool, bool*, convert_bool);
TYPECAST(char, char*, convert_char);
TYPECAST(short, short*, convert_short);
TYPECAST(int, int*, convert_int);
TYPECAST(long, long*, convert_long);
TYPECAST(uchar, uchar*, convert_uchar);
TYPECAST(ushort, ushort*, convert_ushort);
TYPECAST(uint, uint*, convert_uint);
TYPECAST(ulong, ulong*, convert_ulong);
TYPECAST(half, half*, convert_half);
TYPECAST(float, float*, convert_float);
TYPECAST(double, double*, convert_double);
TYPECAST(bfloat16, ushort*, convert_bfloat16);
TYPECAST(complex32, half2*, convert_complex32);
TYPECAST(complex64, float2*, convert_complex64);
TYPECAST(complex128, double2*, convert_complex128);

TYPECASTCONTIGUOUS(bool, bool*, convert_bool);
TYPECASTCONTIGUOUS(char, char*, convert_char);
TYPECASTCONTIGUOUS(short, short*, convert_short);
TYPECASTCONTIGUOUS(int, int*, convert_int);
TYPECASTCONTIGUOUS(long, long*, convert_long);
TYPECASTCONTIGUOUS(uchar, uchar*, convert_uchar);
TYPECASTCONTIGUOUS(ushort, ushort*, convert_ushort);
TYPECASTCONTIGUOUS(uint, uint*, convert_uint);
TYPECASTCONTIGUOUS(ulong, ulong*, convert_ulong);
TYPECASTCONTIGUOUS(half, half*, convert_half);
TYPECASTCONTIGUOUS(float, float*, convert_float);
TYPECASTCONTIGUOUS(double, double*, convert_double);
TYPECASTCONTIGUOUS(bfloat16, ushort*, convert_bfloat16);
TYPECASTCONTIGUOUS(complex32, half2*, convert_complex32);
TYPECASTCONTIGUOUS(complex64, float2*, convert_complex64);
TYPECASTCONTIGUOUS(complex128, double2*, convert_complex128);
