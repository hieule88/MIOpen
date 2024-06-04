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
#ifndef MIOPEN_COSINEEMBEDDINGLOSS_HPP_
#define MIOPEN_COSINEEMBEDDINGLOSS_HPP_

#include <miopen/common.hpp>

namespace miopen {

struct Handle;
struct TensorDescriptor;

size_t GetCosineEmbeddingLossUnreducedForwardWorkspaceSize(Handle& handle,
                                                           const TensorDescriptor input1Desc,
                                                           const TensorDescriptor input2Desc,
                                                           const TensorDescriptor targetDesc,
                                                           const TensorDescriptor outputDesc,
                                                           const float margin);

size_t GetCosineEmbeddingLossReducedForwardWorkspaceSize(Handle& handle,
                                                         const TensorDescriptor input1Desc,
                                                         const TensorDescriptor input2Desc,
                                                         const TensorDescriptor targetDesc,
                                                         const TensorDescriptor outputDesc,
                                                         const float margin);

miopenStatus_t CosineEmbeddingLossUnreducedForward(Handle& handle,
                                                   Data_t workspace,
                                                   size_t workspaceSizeInBytes,
                                                   const TensorDescriptor& input1Desc,
                                                   ConstData_t input1,
                                                   const TensorDescriptor& input2Desc,
                                                   ConstData_t input2,
                                                   const TensorDescriptor& targetDesc,
                                                   ConstData_t target,
                                                   const TensorDescriptor& outputDesc,
                                                   Data_t output,
                                                   const float margin);

miopenStatus_t CosineEmbeddingLossReducedForward(Handle& handle,
                                                 Data_t workspace,
                                                 size_t workspaceSizeInBytes,
                                                 const TensorDescriptor& input1Desc,
                                                 ConstData_t input1,
                                                 const TensorDescriptor& input2Desc,
                                                 ConstData_t input2,
                                                 const TensorDescriptor& targetDesc,
                                                 ConstData_t target,
                                                 const TensorDescriptor& outputDesc,
                                                 Data_t output,
                                                 const float margin,
                                                 const miopenLossReductionMode_t reduction);

size_t GetCosineEmbeddingLossBackwardWorkspaceSize(Handle& handle,
                                                   const TensorDescriptor input1Desc,
                                                   const TensorDescriptor input2Desc,
                                                   const TensorDescriptor targetDesc,
                                                   const TensorDescriptor outputGradDesc,
                                                   const TensorDescriptor input1GradDesc,
                                                   const TensorDescriptor input2GradDesc,
                                                   const float margin);

miopenStatus_t CosineEmbeddingLossUnreducedBackward(Handle& handle,
                                                    Data_t workspace,
                                                    size_t workspaceSizeInBytes,
                                                    const TensorDescriptor& input1Desc,
                                                    ConstData_t input1,
                                                    const TensorDescriptor& input2Desc,
                                                    ConstData_t input2,
                                                    const TensorDescriptor& targetDesc,
                                                    ConstData_t target,
                                                    const TensorDescriptor& outputGradDesc,
                                                    ConstData_t output_grad,
                                                    const TensorDescriptor& input1GradDesc,
                                                    Data_t input1_grad,
                                                    const TensorDescriptor& input2GradDesc,
                                                    Data_t input2_grad,
                                                    const float margin);

miopenStatus_t CosineEmbeddingLossReducedBackward(Handle& handle,
                                                  Data_t workspace,
                                                  size_t workspaceSizeInBytes,
                                                  const TensorDescriptor& input1Desc,
                                                  ConstData_t input1,
                                                  const TensorDescriptor& input2Desc,
                                                  ConstData_t input2,
                                                  const TensorDescriptor& targetDesc,
                                                  ConstData_t target,
                                                  const TensorDescriptor& outputGradDesc,
                                                  ConstData_t output_grad,
                                                  const TensorDescriptor& input1GradDesc,
                                                  Data_t input1_grad,
                                                  const TensorDescriptor& input2GradDesc,
                                                  Data_t input2_grad,
                                                  const float margin,
                                                  const miopenLossReductionMode_t reduction);

} // namespace miopen
#endif // _MIOPEN_COSINEEMBEDDINGLOSS_HPP_
