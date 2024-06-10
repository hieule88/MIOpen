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

#include "miopen/miopen.h"
#include <miopen/softmaxcrossentropywithlogits.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

inline std::ostream& operator<<(std::ostream& os, const std::vector<size_t>& v)
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

static void LogCmdSoftmaxCrossEntropyWithLogits(const miopenTensorDescriptor_t inputDesc,
                                                bool is_fwd,
                                                const miopenLossContiguousMode_t is_contiguous)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        auto dtype = miopen::deref(inputDesc).GetType();
        if(dtype == miopenHalf)
        {
            ss << "softmaxcrossentropywithlogitsfp16";
        }
        else if(dtype == miopenFloat)
        {
            ss << "softmaxcrossentropywithlogits";
        }
        else if(dtype == miopenBFloat16)
        {
            ss << "softmaxcrossentropywithlogitsbfp16";
        }

        MIOPEN_LOG_FUNCTION(inputDesc, is_fwd, is_contiguous);
        ss << " -D " << miopen::deref(inputDesc).GetLengths();
        ss << " -Si " << miopen::deref(inputDesc).GetStrides();

        ss << " -F " << ((is_fwd) ? "1" : "2");
        ss << " -C " << is_contiguous;

        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t miopenGetSoftmaxCrossEntropyWithLogitsForwardWorkspaceSize(
    miopenHandle_t handle,
    const miopenTensorDescriptor_t inputDesc,
    const miopenTensorDescriptor_t targetDesc,
    const miopenTensorDescriptor_t outputDesc,
    const miopenTensorDescriptor_t backpropDesc,
    size_t* sizeInBytes,
    const miopenLossContiguousMode_t is_contiguous)
{

    MIOPEN_LOG_FUNCTION(
        handle, inputDesc, targetDesc, outputDesc, backpropDesc, sizeInBytes, is_contiguous);

    return miopen::try_([&] {
        miopen::deref(sizeInBytes) = miopen::GetSoftmaxCrossEntropyWithLogitsForwardWorkspaceSize(
            miopen::deref(handle),
            miopen::deref(inputDesc),
            miopen::deref(targetDesc),
            miopen::deref(outputDesc),
            miopen::deref(backpropDesc),
            is_contiguous);
    });
}

extern "C" miopenStatus_t
miopenSoftmaxCrossEntropyWithLogitsForward(miopenHandle_t handle,
                                           void* workspace,
                                           size_t workspaceSizeInBytes,
                                           const miopenTensorDescriptor_t inputDesc,
                                           const void* input,
                                           const miopenTensorDescriptor_t targetDesc,
                                           const void* target,
                                           const miopenTensorDescriptor_t outputDesc,
                                           void* output,
                                           const miopenTensorDescriptor_t backpropDesc,
                                           void* backprop,
                                           const miopenLossContiguousMode_t is_contiguous)
{
    MIOPEN_LOG_FUNCTION(handle,
                        workspace,
                        workspaceSizeInBytes,
                        inputDesc,
                        input,
                        targetDesc,
                        target,
                        outputDesc,
                        output,
                        backpropDesc,
                        backprop,
                        is_contiguous);

    LogCmdSoftmaxCrossEntropyWithLogits(inputDesc, true, is_contiguous);

    return miopen::try_([&] {
        miopen::SoftmaxCrossEntropyWithLogitsForward(miopen::deref(handle),
                                                     DataCast(workspace),
                                                     workspaceSizeInBytes,
                                                     miopen::deref(inputDesc),
                                                     DataCast(input),
                                                     miopen::deref(targetDesc),
                                                     DataCast(target),
                                                     miopen::deref(outputDesc),
                                                     DataCast(output),
                                                     miopen::deref(backpropDesc),
                                                     DataCast(backprop),
                                                     is_contiguous);
    });
}

extern "C" miopenStatus_t miopenGetSoftmaxCrossEntropyWithLogitsBackwardWorkspaceSize(
    miopenHandle_t handle,
    const miopenTensorDescriptor_t outputGradDesc,
    const miopenTensorDescriptor_t backpropDesc,
    const miopenTensorDescriptor_t inputDesc,
    const miopenTensorDescriptor_t inputGradDesc,
    const miopenTensorDescriptor_t targetGradDesc,
    size_t* sizeInBytes,
    const miopenLossContiguousMode_t is_contiguous)
{

    MIOPEN_LOG_FUNCTION(handle,
                        outputGradDesc,
                        backpropDesc,
                        inputDesc,
                        inputGradDesc,
                        targetGradDesc,
                        sizeInBytes,
                        is_contiguous);

    return miopen::try_([&] {
        miopen::deref(sizeInBytes) = miopen::GetSoftmaxCrossEntropyWithLogitsBackwardWorkspaceSize(
            miopen::deref(handle),
            miopen::deref(outputGradDesc),
            miopen::deref(backpropDesc),
            miopen::deref(inputDesc),
            miopen::deref(inputGradDesc),
            miopen::deref(targetGradDesc),
            is_contiguous);
    });
}

extern "C" miopenStatus_t
miopenSoftmaxCrossEntropyWithLogitsBackward(miopenHandle_t handle,
                                            void* workspace,
                                            size_t workspaceSizeInBytes,
                                            const miopenTensorDescriptor_t outputGradDesc,
                                            const void* output_grad,
                                            const miopenTensorDescriptor_t backpropDesc,
                                            const void* backprop,
                                            const miopenTensorDescriptor_t inputDesc,
                                            const void* input,
                                            const miopenTensorDescriptor_t inputGradDesc,
                                            void* input_grad,
                                            const miopenTensorDescriptor_t targetGradDesc,
                                            void* target_grad,
                                            const miopenLossContiguousMode_t is_contiguous)
{
    MIOPEN_LOG_FUNCTION(handle,
                        workspace,
                        workspaceSizeInBytes,
                        outputGradDesc,
                        output_grad,
                        backpropDesc,
                        backprop,
                        inputDesc,
                        input,
                        inputGradDesc,
                        input_grad,
                        targetGradDesc,
                        target_grad,
                        is_contiguous);

    LogCmdSoftmaxCrossEntropyWithLogits(inputDesc, false, is_contiguous);

    return miopen::try_([&] {
        miopen::SoftmaxCrossEntropyWithLogitsBackward(miopen::deref(handle),
                                                      DataCast(workspace),
                                                      workspaceSizeInBytes,
                                                      miopen::deref(outputGradDesc),
                                                      DataCast(output_grad),
                                                      miopen::deref(backpropDesc),
                                                      DataCast(backprop),
                                                      miopen::deref(inputDesc),
                                                      DataCast(input),
                                                      miopen::deref(inputGradDesc),
                                                      DataCast(input_grad),
                                                      miopen::deref(targetGradDesc),
                                                      DataCast(target_grad),
                                                      is_contiguous);
    });
}
