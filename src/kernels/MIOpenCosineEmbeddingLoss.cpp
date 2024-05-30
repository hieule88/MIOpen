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

template <typename TI, typename TO>
__device__ void cosineembeddinglossNorm2d(const TI* __restrict__ input1,
                                          const TI* __restrict__ input2,
                                          TO* __restrict__ workspace,
                                          tensor_view_2d_t input1_tv,
                                          tensor_view_2d_t input2_tv)
{
    uint64_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    size_t N = input1_tv.size[0], D = input1_tv.size[1];
    size_t n[2];
    GET_ND(n[0], n[1], gid, input1_tv)
    if(!(n[0] < N))
        return;

    size_t I1idx = TV2D_IDX(input1_tv, n[0], n[1]);
    size_t I2idx = TV2D_IDX(input2_tv, n[0], n[1]);

    FLOAT_ACCUM cos_term = CVT_FLOAT2ACCUM(input1[I1idx]) * CVT_FLOAT2ACCUM(input2[I2idx]);
    FLOAT_ACCUM norm1    = CVT_FLOAT2ACCUM(input1[I1idx]) * CVT_FLOAT2ACCUM(input1[I1idx]);
    FLOAT_ACCUM norm2    = CVT_FLOAT2ACCUM(input2[I2idx]) * CVT_FLOAT2ACCUM(input2[I2idx]);

    size_t sum_size = N * D;

    workspace[0 * sum_size + gid] = CVT_ACCUM2FLOAT(cos_term);
    workspace[1 * sum_size + gid] = CVT_ACCUM2FLOAT(norm1);
    workspace[2 * sum_size + gid] = CVT_ACCUM2FLOAT(norm2);
}
extern "C" __global__ void CosineEmbeddingLossNorm2d(const INPUT_TYPE* __restrict__ input1,
                                                     const INPUT_TYPE* __restrict__ input2,
                                                     D_TYPE* __restrict__ workspace,
                                                     tensor_view_2d_t input1_tv,
                                                     tensor_view_2d_t input2_tv)
{
    cosineembeddinglossNorm2d<INPUT_TYPE, D_TYPE>(input1, input2, workspace, input1_tv, input2_tv);
}

template <typename TI, typename TO>
__device__ void cosineembeddinglossUnreducedForward2d(const TI* __restrict__ workspace,
                                                      const int32_t* __restrict__ target,
                                                      TO* __restrict__ output,
                                                      float margin,
                                                      tensor_view_1d_t target_tv,
                                                      tensor_view_1d_t output_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    size_t N = target_tv.size[0];
    size_t n = gid;
    if(!(n < N))
        return;

    FLOAT_ACCUM cos_term = workspace[n + 0 * N];
    FLOAT_ACCUM norm1    = workspace[n + 1 * N];
    FLOAT_ACCUM norm2    = workspace[n + 2 * N];
    norm1                = sqrt(norm1);
    norm2                = sqrt(norm2);
    cos_term /= norm1 * norm2;

    size_t Tidx      = TV1D_IDX(target_tv, n);
    int32_t t        = target[Tidx];
    FLOAT_ACCUM loss = 0.0f;
    if(t == 1)
        loss = 1.0f - cos_term;
    else
        loss = max(0.0f, cos_term - margin);

    size_t Oidx  = TV1D_IDX(output_tv, n);
    output[Oidx] = CVT_ACCUM2FLOAT(loss);
}

extern "C" __global__ void
CosineEmbeddingLossUnreducedForward2d(const D_TYPE* __restrict__ workspace,
                                      const int32_t* __restrict__ target,
                                      OUTPUT_TYPE* __restrict__ output,
                                      float margin,
                                      tensor_view_1d_t target_tv,
                                      tensor_view_1d_t output_tv)
{
    cosineembeddinglossUnreducedForward2d<D_TYPE, OUTPUT_TYPE>(
        workspace, target, output, margin, target_tv, output_tv);
}

template <typename TI, typename TO>
__device__ void cosineembeddinglossReducedForward2d(const TI* __restrict__ workspace,
                                                    const int32_t* __restrict__ target,
                                                    TO* __restrict__ loss_sum,
                                                    float margin,
                                                    float divisor,
                                                    tensor_view_1d_t target_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    size_t N = target_tv.size[0];
    size_t n = gid;
    if(!(n < N))
        return;

    FLOAT_ACCUM cos_term = workspace[n + 0 * N];
    FLOAT_ACCUM norm1    = workspace[n + 1 * N];
    FLOAT_ACCUM norm2    = workspace[n + 2 * N];
    norm1                = sqrt(norm1);
    norm2                = sqrt(norm2);
    cos_term /= norm1 * norm2;

    size_t Tidx      = TV1D_IDX(target_tv, n);
    int32_t t        = target[Tidx];
    FLOAT_ACCUM loss = 0.0f;
    if(t == 1)
        loss = 1.0f - cos_term;
    else
        loss = max(0.0f, cos_term - margin);

    loss_sum[n] = CVT_ACCUM2FLOAT(loss / divisor);
}

extern "C" __global__ void CosineEmbeddingLossReducedForward2d(const D_TYPE* __restrict__ workspace,
                                                               const int32_t* __restrict__ target,
                                                               OUTPUT_TYPE* __restrict__ loss_sum,
                                                               float margin,
                                                               float divisor,
                                                               tensor_view_1d_t target_tv)
{
    cosineembeddinglossReducedForward2d<D_TYPE, OUTPUT_TYPE>(
        workspace, target, loss_sum, margin, divisor, target_tv);
}

template <typename TI, typename TO, typename T>
__device__ void cosineembeddinglossUnreducedBackward2d(const T* __restrict__ workspace,
                                                       const TI* __restrict__ input1,
                                                       const TI* __restrict__ input2,
                                                       const int32_t* __restrict__ target,
                                                       const TI* __restrict__ output_grad,
                                                       TO* __restrict__ input1_grad,
                                                       TO* __restrict__ input2_grad,
                                                       float margin,
                                                       tensor_view_2d_t input1_tv,
                                                       tensor_view_2d_t input2_tv,
                                                       tensor_view_1d_t target_tv,
                                                       tensor_view_1d_t output_grad_tv,
                                                       tensor_view_2d_t input1_grad_tv,
                                                       tensor_view_2d_t input2_grad_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    size_t n[2];
    GET_ND(n[0], n[1], gid, input1_tv)

    size_t N = input1_tv.size[0], D = input1_tv.size[1];
    if(!(n[0] < N))
        return;

    FLOAT_ACCUM cos_term = CVT_FLOAT2ACCUM(workspace[0 * N + n[0]]);
    FLOAT_ACCUM norm1    = CVT_FLOAT2ACCUM(workspace[1 * N + n[0]]);
    FLOAT_ACCUM norm2    = CVT_FLOAT2ACCUM(workspace[2 * N + n[0]]);
    norm1                = sqrt(norm1);
    norm2                = sqrt(norm2);
    cos_term /= norm1 * norm2;

    size_t dOidx   = TV1D_IDX(output_grad_tv, n[0]);
    FLOAT_ACCUM og = CVT_FLOAT2ACCUM(output_grad[dOidx]);

    size_t Tidx = TV1D_IDX(target_tv, n[0]);
    int32_t t   = target[Tidx];

    size_t I1idx = TV2D_IDX(input1_tv, n[0], n[1]);
    size_t I2idx = TV2D_IDX(input2_tv, n[0], n[1]);

    FLOAT_ACCUM i1              = CVT_FLOAT2ACCUM(input1[I1idx]);
    FLOAT_ACCUM i2              = CVT_FLOAT2ACCUM(input2[I2idx]);
    FLOAT_ACCUM input1_grad_val = 0.0f;
    FLOAT_ACCUM input2_grad_val = 0.0f;

    if(t == 1)
    {
        if(input1_grad)
        {
            input1_grad_val = -(i2 / (norm1 * norm2) - cos_term * i1 / (norm1 * norm1));
        }
        if(input2_grad)
        {
            input2_grad_val = -(i1 / (norm1 * norm2) - cos_term * i2 / (norm2 * norm2));
        }
    }
    else
    {
        if(cos_term - margin < 0.0f)
            return;
        if(input1_grad)
        {
            input1_grad_val = i2 / (norm1 * norm2) - cos_term * i1 / (norm1 * norm1);
        }
        if(input2_grad)
        {
            input2_grad_val = i1 / (norm1 * norm2) - cos_term * i2 / (norm2 * norm2);
        }
    }
    if(input1_grad)
    {
        size_t IG1idx       = TV2D_IDX(input1_grad_tv, n[0], n[1]);
        input1_grad[IG1idx] = CVT_ACCUM2FLOAT(input1_grad_val * og);
    }
    if(input2_grad)
    {
        size_t IG2idx       = TV2D_IDX(input2_grad_tv, n[0], n[1]);
        input2_grad[IG2idx] = CVT_ACCUM2FLOAT(input2_grad_val * og);
    }
}

extern "C" __global__ void
CosineEmbeddingLossUnreducedBackward2d(const D_TYPE* __restrict__ workspace,
                                       const INPUT_TYPE* __restrict__ input1,
                                       const INPUT_TYPE* __restrict__ input2,
                                       const int32_t* __restrict__ target,
                                       const INPUT_TYPE* __restrict__ output_grad,
                                       OUTPUT_TYPE* __restrict__ input1_grad,
                                       OUTPUT_TYPE* __restrict__ input2_grad,
                                       float margin,
                                       tensor_view_2d_t input1_tv,
                                       tensor_view_2d_t input2_tv,
                                       tensor_view_1d_t target_tv,
                                       tensor_view_1d_t output_grad_tv,
                                       tensor_view_2d_t input1_grad_tv,
                                       tensor_view_2d_t input2_grad_tv)
{
    cosineembeddinglossUnreducedBackward2d<INPUT_TYPE, OUTPUT_TYPE, D_TYPE>(workspace,
                                                                            input1,
                                                                            input2,
                                                                            target,
                                                                            output_grad,
                                                                            input1_grad,
                                                                            input2_grad,
                                                                            margin,
                                                                            input1_tv,
                                                                            input2_tv,
                                                                            target_tv,
                                                                            output_grad_tv,
                                                                            input1_grad_tv,
                                                                            input2_grad_tv);
}

template <typename TI, typename TO, typename T>
__device__ void cosineembeddinglossReducedBackward2d(const T* __restrict__ workspace,
                                                     const TI* __restrict__ input1,
                                                     const TI* __restrict__ input2,
                                                     const int32_t* __restrict__ target,
                                                     const TI* __restrict__ output_grad,
                                                     TO* __restrict__ input1_grad,
                                                     TO* __restrict__ input2_grad,
                                                     float margin,
                                                     float divisor,
                                                     tensor_view_2d_t input1_tv,
                                                     tensor_view_2d_t input2_tv,
                                                     tensor_view_1d_t target_tv,
                                                     tensor_view_1d_t output_grad_tv,
                                                     tensor_view_2d_t input1_grad_tv,
                                                     tensor_view_2d_t input2_grad_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    size_t n[2];
    GET_ND(n[0], n[1], gid, input1_tv)

    size_t N = input1_tv.size[0], D = input1_tv.size[1];
    if(!(n[0] < N))
        return;

    FLOAT_ACCUM cos_term = CVT_FLOAT2ACCUM(workspace[0 * N + n[0]]);
    FLOAT_ACCUM norm1    = CVT_FLOAT2ACCUM(workspace[1 * N + n[0]]);
    FLOAT_ACCUM norm2    = CVT_FLOAT2ACCUM(workspace[2 * N + n[0]]);
    norm1                = sqrt(norm1);
    norm2                = sqrt(norm2);
    cos_term /= norm1 * norm2;

    size_t dOidx   = TV1D_IDX(output_grad_tv, 0);
    FLOAT_ACCUM og = CVT_FLOAT2ACCUM(output_grad[dOidx]);

    size_t Tidx = TV1D_IDX(target_tv, n[0]);
    int32_t t   = target[Tidx];

    size_t I1idx = TV2D_IDX(input1_tv, n[0], n[1]);
    size_t I2idx = TV2D_IDX(input2_tv, n[0], n[1]);

    FLOAT_ACCUM i1              = CVT_FLOAT2ACCUM(input1[I1idx]);
    FLOAT_ACCUM i2              = CVT_FLOAT2ACCUM(input2[I2idx]);
    FLOAT_ACCUM input1_grad_val = 0.0f;
    FLOAT_ACCUM input2_grad_val = 0.0f;

    if(t == 1)
    {
        if(input1_grad)
        {
            input1_grad_val = -(i2 / (norm1 * norm2) - cos_term * i1 / (norm1 * norm1));
        }
        if(input2_grad)
        {
            input2_grad_val = -(i1 / (norm1 * norm2) - cos_term * i2 / (norm2 * norm2));
        }
    }
    else
    {
        if(cos_term - margin < 0.0f)
            return;
        if(input1_grad)
        {
            input1_grad_val = i2 / (norm1 * norm2) - cos_term * i1 / (norm1 * norm1);
        }
        if(input2_grad)
        {
            input2_grad_val = i1 / (norm1 * norm2) - cos_term * i2 / (norm2 * norm2);
        }
    }
    if(input1_grad)
    {
        size_t IG1idx       = TV2D_IDX(input1_grad_tv, n[0], n[1]);
        input1_grad[IG1idx] = CVT_ACCUM2FLOAT(input1_grad_val * og / divisor);
    }
    if(input2_grad)
    {
        size_t IG2idx       = TV2D_IDX(input2_grad_tv, n[0], n[1]);
        input2_grad[IG2idx] = CVT_ACCUM2FLOAT(input2_grad_val * og / divisor);
    }
}

extern "C" __global__ void
CosineEmbeddingLossReducedBackward2d(const D_TYPE* __restrict__ workspace,
                                     const INPUT_TYPE* __restrict__ input1,
                                     const INPUT_TYPE* __restrict__ input2,
                                     const int32_t* __restrict__ target,
                                     const INPUT_TYPE* __restrict__ output_grad,
                                     OUTPUT_TYPE* __restrict__ input1_grad,
                                     OUTPUT_TYPE* __restrict__ input2_grad,
                                     float margin,
                                     float divisor,
                                     tensor_view_2d_t input1_tv,
                                     tensor_view_2d_t input2_tv,
                                     tensor_view_1d_t target_tv,
                                     tensor_view_1d_t output_grad_tv,
                                     tensor_view_2d_t input1_grad_tv,
                                     tensor_view_2d_t input2_grad_tv)
{
    cosineembeddinglossReducedBackward2d<INPUT_TYPE, OUTPUT_TYPE, D_TYPE>(workspace,
                                                                          input1,
                                                                          input2,
                                                                          target,
                                                                          output_grad,
                                                                          input1_grad,
                                                                          input2_grad,
                                                                          margin,
                                                                          divisor,
                                                                          input1_tv,
                                                                          input2_tv,
                                                                          target_tv,
                                                                          output_grad_tv,
                                                                          input1_grad_tv,
                                                                          input2_grad_tv);
}
