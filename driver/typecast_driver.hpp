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

#include "InputFlags.hpp"
#include "driver.hpp"
#include "mloTypeCastHost.hpp"
#include "random.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"

#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>

#include <miopen/env.hpp>
#include <miopen/handle.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <vector>

template <typename Tin, typename Tout>
class TypeCastDriver : public Driver
{
public:
    TypeCastDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&outputDesc);

        data_type = miopen_type<Tin>{};
    }

    std::vector<int> ComputeStrides(std::vector<int> input);
    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override;
    int RunBackwardCPU();

    Tout GetTolerance();
    int VerifyBackward() override;
    int VerifyForward() override;
    ~TypeCastDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(outputDesc);
    }

private:
    InputFlags inflags;
    int forw;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t outputDesc;

    std::unique_ptr<GPUMem> input_dev;
    std::unique_ptr<GPUMem> output_dev;

    std::vector<Tin> input;
    std::vector<Tout> output;
    std::vector<Tout> output_host;

    uint64_t bits_to_truncate;
    std::vector<int> in_len;
    bool isContiguous;
};

template <typename Tin, typename Tout>
int TypeCastDriver<Tin, Tout>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);
    forw         = inflags.GetValueInt("forw");
    isContiguous = inflags.GetValueInt("is-contiguous") == 1 ? true : false;

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }

    return miopenStatusSuccess;
}

template <typename Tin, typename Tout>
int TypeCastDriver<Tin, Tout>::GetandSetData()
{
    in_len           = inflags.GetValueTensor("input_dim").lengths;
    bits_to_truncate = inflags.GetValueUint64("bits_to_truncate");

    std::vector<int> in_stride = ComputeStrides(in_len);

    if(SetTensorNd(inputDesc, in_len, in_stride, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing input tensor: " + inflags.GetValueStr("input_dim") + ".");
    if(SetTensorNd(outputDesc, in_len, miopen_type<Tout>{}) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing output tensor.");

    return miopenStatusSuccess;
}

// Equivalent to: tensor.tranpose(0, -1).contiguous().tranpose(0, -1) incase contiguous = False
template <typename Tin, typename Tout>
std::vector<int> TypeCastDriver<Tin, Tout>::ComputeStrides(std::vector<int> inputDim)
{
    if(!isContiguous)
        std::swap(inputDim.front(), inputDim.back());
    std::vector<int> strides(inputDim.size());
    strides.back() = 1;
    for(int i = inputDim.size() - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * inputDim[i + 1];
    if(!isContiguous)
        std::swap(strides.front(), strides.back());
    return strides;
}

template <typename Tin, typename Tout>
int TypeCastDriver<Tin, Tout>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward TypeCast (Default=1)", "int");
    inflags.AddTensorFlag("input_dim",
                          'd',
                          "7x9x10x10x10",
                          "The dimensional lengths of the input tensors: NxCxDxHxW. "
                          "Example: 7x9x10x10x10.");
    inflags.AddInputFlag(
        "bits_to_truncate", 'b', "0", "Number of bits to truncate (Default=0)", "int");

    inflags.AddInputFlag("is-contiguous", 'C', "1", "is-contiguous (Default=1)", "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "1", "Time (Default=1)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tin, typename Tout>
int TypeCastDriver<Tin, Tout>::AllocateBuffersAndCopy()
{
    size_t input_sz = GetTensorSize(inputDesc);

    uint32_t ctx = 0;

    input_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, input_sz, sizeof(Tin)));
    output_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, input_sz, sizeof(Tout)));

    input       = std::vector<Tin>(input_sz, static_cast<Tin>(0));
    output      = std::vector<Tout>(input_sz, static_cast<Tout>(0));
    output_host = std::vector<Tout>(input_sz, static_cast<Tout>(0));

    for(size_t i = 0; i < input_sz; i++)
    {
        input[i] = prng::gen_A_to_B<Tin>(static_cast<Tin>(-10.0), static_cast<Tin>(10.0));
    }

    if(input_dev->ToGPU(GetStream(), input.data()) != 0)
    {
        std::cerr << "Error copying (input) to GPU, size: " << input_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }

    if(output_dev->ToGPU(GetStream(), output.data()) != 0)
    {
        std::cerr << "Error copying (output) to GPU, size: " << output_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tin, typename Tout>
int TypeCastDriver<Tin, Tout>::RunForwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenStatus_t status = miopenStatusSuccess;
        status                = miopenTypeCast(GetHandle(),
                                inputDesc,
                                input_dev->GetMem(),
                                outputDesc,
                                output_dev->GetMem(),
                                bits_to_truncate);

        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenTypeCast");

        float time = 0.0;
        miopenGetKernelTime(GetHandle(), &time);
        kernel_total_time += time;
        if(i == 0)
            kernel_first_time = time;
    }

    if(inflags.GetValueInt("time") == 1)
    {
        STOP_TIME
        int iter = inflags.GetValueInt("iter");
        if(WALL_CLOCK)
            std::cout << "Wall-clock Time Forward TypeCast Elapsed: " << t.gettime_ms() / iter
                      << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward TypeCast Elapsed: " << kernel_average_time << " ms"
                  << std::endl;
    }

    if(output_dev->FromGPU(GetStream(), output.data()) != 0)
    {
        std::cerr << "Error copying (output_dev) from GPU, size: " << output_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tin, typename Tout>
int TypeCastDriver<Tin, Tout>::RunForwardCPU()
{
    int status = miopenStatusSuccess;

    status = mloTypeCastRunHost<Tin, Tout>(
        inputDesc, input.data(), outputDesc, output_host.data(), bits_to_truncate);
    MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in mloTypeCastRunHost");

    return status;
}

template <typename Tin, typename Tout>
int TypeCastDriver<Tin, Tout>::RunBackwardGPU()
{
    return miopenStatusNotImplemented;
}

template <typename Tin, typename Tout>
int TypeCastDriver<Tin, Tout>::RunBackwardCPU()
{
    return miopenStatusNotImplemented;
}

template <typename Tin, typename Tout>
Tout TypeCastDriver<Tin, Tout>::GetTolerance()
{
    Tout tolerance = static_cast<Tout>(std::numeric_limits<Tout>::epsilon() * 10);
    return tolerance;
}

template <typename Tin, typename Tout>
int TypeCastDriver<Tin, Tout>::VerifyForward()
{
    RunForwardCPU();
    const Tout tolerance = GetTolerance();

    for(int i = 0; i < 10; ++i)
    {
        std::cout << "Input: " << input[i] << " GPU: " << output[i] << " CPU: " << output_host[i]
                  << std::endl;
    }

    auto error = miopen::rms_range(output_host, output);
    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward TypeCast Output FAILED: " << error << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward TypeCast Output Verifies on CPU and GPU (err=" << error << ")"
                  << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tin, typename Tout>
int TypeCastDriver<Tin, Tout>::VerifyBackward()
{
    return miopenStatusNotImplemented;
}
