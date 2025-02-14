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
#pragma once

#include <miopen/tensor.hpp>
#include <miopen/tensor_view_utils.hpp>
#include <../test/ford.hpp>

template <typename DTYPE>
inline DTYPE truncate_func(DTYPE val, uint64_t nbits)
{
    if constexpr(std::is_same_v<DTYPE, float>)
    {
        uint32_t intVal;
        memcpy(&intVal, &val, sizeof(float));
        intVal &= (0xFFFFFFFFu << nbits);
        memcpy(&val, &intVal, sizeof(float));
        return val;
    }
    else if constexpr(std::is_same_v<DTYPE, double>)
    {
        uint64_t intVal;
        memcpy(&intVal, &val, sizeof(double));
        intVal &= (0xFFFFFFFFFFFFFFFFull << nbits);
        memcpy(&val, &intVal, sizeof(double));
        return val;
    }
    else
    {
        return val;
    }
}

template <class TI, class TO>
inline TO cast_func(TI val)
{
    return static_cast<TO>(val);
}

template <typename TI, typename TO>
int32_t mloTypeCastRunHost(const miopenTensorDescriptor_t inputDesc,
                           const TI* input,
                           const miopenTensorDescriptor_t outputDesc,
                           TO* output,
                           const uint64_t bits_to_truncate)
{
    auto input_tv  = miopen::get_inner_expanded_tv<5>(miopen::deref(inputDesc));
    auto output_tv = miopen::get_inner_expanded_tv<5>(miopen::deref(outputDesc));

    par_ford(miopen::deref(outputDesc).GetElementSize())([&](auto gid) {
        tensor_layout_t<5> layout(input_tv, gid);
        TI val = input[input_tv.get_tensor_view_idx(layout)];
        if(bits_to_truncate)
        {
            val = truncate_func<TI>(val, bits_to_truncate);
        }
        output[output_tv.get_tensor_view_idx(layout)] = cast_func<TI, TO>(val);
    });

    return miopenStatusSuccess;
}
