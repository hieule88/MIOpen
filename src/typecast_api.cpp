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

#include <miopen/typecast.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

inline std::ostream& operator<<(std::ostream& os, const std::vector<uint64_t>& v)
{
    os << '{';
    for(int i = 0; i < v.size(); ++i)
    {
        if(i != 0)
            os << ',';
        os << v[i];
    }
    os << '}';
    return os;
}

inline void LogCmdTypeCast(const miopenTensorDescriptor_t xDesc,
                           const miopenTensorDescriptor_t yDesc)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        auto x_dtype = miopen::deref(xDesc).GetType();
        auto y_dtype = miopen::deref(yDesc).GetType();

        if(x_dtype == miopenHalf)
        {
            ss << "typecastfp16_";
        }
        else if(x_dtype == miopenFloat)
        {
            ss << "typecastfp32_";
        }
        else if(x_dtype == miopenInt32)
        {
            ss << "typecastint32_";
        }
        else if(x_dtype == miopenInt8)
        {
            ss << "typecastint8_";
        }
        else if(x_dtype == miopenBFloat16)
        {
            ss << "typecastbfp16_";
        }
        else if(x_dtype == miopenDouble)
        {
            ss << "typecastfp64_";
        }
        else if(x_dtype == miopenInt64)
        {
            ss << "typecastint64_";
        }

        if(y_dtype == miopenHalf)
        {
            ss << "to_fp16";
        }
        else if(y_dtype == miopenFloat)
        {
            ss << "to_fp32";
        }
        else if(y_dtype == miopenInt32)
        {
            ss << "to_int32";
        }
        else if(y_dtype == miopenInt8)
        {
            ss << "to_int8";
        }
        else if(y_dtype == miopenBFloat16)
        {
            ss << "to_bfp16";
        }
        else if(y_dtype == miopenDouble)
        {
            ss << "to_fp64";
        }
        else if(y_dtype == miopenInt64)
        {
            ss << "to_int64";
        }

        ss << " -Xs ";
        ss << miopen::deref(xDesc).GetLengths();
        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t miopenTypeCast(miopenHandle_t handle,
                                         const miopenTensorDescriptor_t inputDesc,
                                         const void* input,
                                         const miopenTensorDescriptor_t outputDesc,
                                         void* output,
                                         const uint64_t bits_to_truncate)
{
    MIOPEN_LOG_FUNCTION(handle, inputDesc, input, outputDesc, output, bits_to_truncate);

    LogCmdTypeCast(inputDesc, outputDesc);

    return miopen::try_([&] {
        miopen::typecast::TypeCast(miopen::deref(handle),
                                   miopen::deref(inputDesc),
                                   DataCast(input),
                                   miopen::deref(outputDesc),
                                   DataCast(output),
                                   bits_to_truncate);
    });
}
