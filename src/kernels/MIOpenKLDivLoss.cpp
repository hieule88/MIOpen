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
__device__ void kldivLossUnreducedBackward5d(const TI* __restrict__ input,
                                             const TI* __restrict__ target,
                                             const TI* __restrict__ output_grad,
                                             TO* __restrict__ input_grad,
                                             TO* __restrict__ target_grad,
                                             bool log_target,
                                             tensor_view_t<5> input_tv,
                                             tensor_view_t<5> target_tv,
                                             tensor_view_t<5> output_grad_tv,
                                             tensor_view_t<5> input_grad_tv,
                                             tensor_view_t<5> target_grad_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    auto tensor_layout = tensor_layout_t<5>(input_grad_tv, gid);

    if(tensor_layout.layout[0] >= input_grad_tv.size[0])
        return;

    size_t Iidx  = input_tv.get_tensor_view_idx(tensor_layout);
    size_t Tidx  = target_tv.get_tensor_view_idx(tensor_layout);
    size_t dOidx = output_grad_tv.get_tensor_view_idx(tensor_layout);
    size_t dIidx = input_grad_tv.get_tensor_view_idx(tensor_layout);
    size_t dTidx = target_grad_tv.get_tensor_view_idx(tensor_layout);

    FLOAT_ACCUM input_value       = CVT_FLOAT2ACCUM(input[Iidx]);
    FLOAT_ACCUM target_value      = CVT_FLOAT2ACCUM(target[Tidx]);
    FLOAT_ACCUM output_grad_value = CVT_FLOAT2ACCUM(output_grad[dOidx]);
    FLOAT_ACCUM forward_output;

    if(log_target)
    {
        FLOAT_ACCUM exp_target = exp(target_value);
        forward_output         = exp_target * (target_value - input_value);
        if(input_grad)
        {
            FLOAT_ACCUM input_grad_value =
                isnan(forward_output) ? CVT_FP32_2ACCUM(0.0f)
                                      : CVT_FP32_2ACCUM(-1.0f) * exp_target * output_grad_value;
            input_grad[dIidx] = CVT_ACCUM2FLOAT(input_grad_value);
        }
        if(target_grad)
        {
            FLOAT_ACCUM target_grad_value = (forward_output + exp_target) * output_grad_value;
            target_grad[dTidx]            = CVT_ACCUM2FLOAT(target_grad_value);
        }
    }
    else
    {
        forward_output = target_value * (log(target_value) - input_value);
        if(input_grad)
        {
            FLOAT_ACCUM input_grad_value =
                isnan(forward_output) ? 0.0f : -target_value * output_grad_value;
            input_grad[dIidx] = CVT_ACCUM2FLOAT(input_grad_value);
        }
        if(target_grad)
        {
            FLOAT_ACCUM target_grad_value =
                (target_value == 0) ? 0.0f
                                    : (1 + (log(target_value) - input_value)) * output_grad_value;
            target_grad[dTidx] = CVT_ACCUM2FLOAT(target_grad_value);
        }
    }
}

extern "C" __global__ void KLDivLossUnreducedBackward5d(const INPUT_TYPE* __restrict__ input,
                                                        const INPUT_TYPE* __restrict__ target,
                                                        const INPUT_TYPE* __restrict__ output_grad,
                                                        OUTPUT_TYPE* __restrict__ input_grad,
                                                        OUTPUT_TYPE* __restrict__ target_grad,
                                                        bool log_target,
                                                        tensor_view_t<5> input_tv,
                                                        tensor_view_t<5> target_tv,
                                                        tensor_view_t<5> output_grad_tv,
                                                        tensor_view_t<5> input_grad_tv,
                                                        tensor_view_t<5> target_grad_tv)
{
    kldivLossUnreducedBackward5d<INPUT_TYPE, OUTPUT_TYPE>(input,
                                                          target,
                                                          output_grad,
                                                          input_grad,
                                                          target_grad,
                                                          log_target,
                                                          input_tv,
                                                          target_tv,
                                                          output_grad_tv,
                                                          input_grad_tv,
                                                          target_grad_tv);
}

template <typename TI, typename TO>
__device__ void kldivLossReducedBackward5d(const TI* __restrict__ input,
                                           const TI* __restrict__ target,
                                           const TI* __restrict__ output_grad,
                                           TO* __restrict__ input_grad,
                                           TO* __restrict__ target_grad,
                                           float divisor,
                                           bool log_target,
                                           tensor_view_t<5> input_tv,
                                           tensor_view_t<5> target_tv,
                                           tensor_view_t<1> output_grad_tv,
                                           tensor_view_t<5> input_grad_tv,
                                           tensor_view_t<5> target_grad_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    auto tensor_layout = tensor_layout_t<5>(input_grad_tv, gid);

    if(tensor_layout.layout[0] >= input_grad_tv.size[0])
        return;

    size_t Iidx  = input_tv.get_tensor_view_idx(tensor_layout);
    size_t Tidx  = target_tv.get_tensor_view_idx(tensor_layout);
    size_t dOidx = output_grad_tv.get_tensor_view_idx({0});
    size_t dIidx = input_grad_tv.get_tensor_view_idx(tensor_layout);
    size_t dTidx = target_grad_tv.get_tensor_view_idx(tensor_layout);

    FLOAT_ACCUM input_value       = CVT_FLOAT2ACCUM(input[Iidx]);
    FLOAT_ACCUM target_value      = CVT_FLOAT2ACCUM(target[Tidx]);
    FLOAT_ACCUM output_grad_value = CVT_FLOAT2ACCUM(output_grad[dOidx]);
    FLOAT_ACCUM forward_output;
    FLOAT_ACCUM d = CVT_FP32_2ACCUM(divisor);

    if(log_target)
    {
        FLOAT_ACCUM exp_target = exp(target_value);
        forward_output         = exp_target * (target_value - input_value);
        if(input_grad)
        {
            FLOAT_ACCUM input_grad_value =
                isnan(forward_output)
                    ? CVT_FP32_2ACCUM(0.0f)
                    : CVT_FP32_2ACCUM(-1.0f) * (exp_target / d) * output_grad_value;
            input_grad[dIidx] = CVT_ACCUM2FLOAT(input_grad_value);
        }
        if(target_grad)
        {
            FLOAT_ACCUM target_grad_value = ((forward_output + exp_target) / d) * output_grad_value;
            target_grad[dTidx]            = CVT_ACCUM2FLOAT(target_grad_value);
        }
    }
    else
    {
        forward_output = target_value * (log(target_value) - input_value);
        if(input_grad)
        {
            FLOAT_ACCUM input_grad_value =
                isnan(forward_output)
                    ? CVT_FP32_2ACCUM(0.0f)
                    : CVT_FP32_2ACCUM(-1.0f) * target_value / d * output_grad_value;
            input_grad[dIidx] = CVT_ACCUM2FLOAT(input_grad_value);
        }
        if(target_grad)
        {
            FLOAT_ACCUM target_grad_value =
                (target_value == CVT_FP32_2ACCUM(0.0f))
                    ? CVT_FP32_2ACCUM(0.0f)
                    : (CVT_FP32_2ACCUM(1.0f) + (log(target_value) - input_value)) / d *
                          output_grad_value;
            target_grad[dTidx] = CVT_ACCUM2FLOAT(target_grad_value);
        }
    }
}

extern "C" __global__ void KLDivLossReducedBackward5d(const INPUT_TYPE* __restrict__ input,
                                                      const INPUT_TYPE* __restrict__ target,
                                                      const INPUT_TYPE* __restrict__ output_grad,
                                                      OUTPUT_TYPE* __restrict__ input_grad,
                                                      OUTPUT_TYPE* __restrict__ target_grad,
                                                      float divisor,
                                                      bool log_target,
                                                      tensor_view_t<5> input_tv,
                                                      tensor_view_t<5> target_tv,
                                                      tensor_view_t<1> output_grad_tv,
                                                      tensor_view_t<5> input_grad_tv,
                                                      tensor_view_t<5> target_grad_tv)
{
    kldivLossReducedBackward5d<INPUT_TYPE, OUTPUT_TYPE>(input,
                                                        target,
                                                        output_grad,
                                                        input_grad,
                                                        target_grad,
                                                        divisor,
                                                        log_target,
                                                        input_tv,
                                                        target_tv,
                                                        output_grad_tv,
                                                        input_grad_tv,
                                                        target_grad_tv);
}
