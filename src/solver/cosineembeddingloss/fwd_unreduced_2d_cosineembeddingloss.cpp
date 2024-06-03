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

#include "miopen/conv_solution.hpp"
#include "miopen/execution_context.hpp"
#include "miopen/invoke_params.hpp"
#include <miopen/cosineembeddingloss/solvers.hpp>
#include <miopen/cosineembeddingloss/utils.hpp>

#include <miopen/cosineembeddingloss/invoke_params.hpp>
#include <miopen/datatype.hpp>
#include <miopen/cosineembeddingloss.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/tensor_view.hpp>

#define LOCAL_SIZE_UNREDUCED_FWD 256
#define LOCAL_SIZE_REDUCED_SUM 256

namespace miopen {

namespace solver {

namespace cosineembeddingloss {

inline void ConstructNormParamsKernels(
    const ExecutionContext& context,
    const miopen::cosineembeddingloss::FwdUnreducedProblemDescription& problem,
    ConvSolution& result,
    const KernelBuildParameters& build_params)
{
    auto input_size = problem.GetInput1Desc().GetElementSize();
    result.construction_params.push_back(make_hip_kernel({LOCAL_SIZE_REDUCED_SUM},
                                                         {input_size},
                                                         "MIOpenCosineEmbeddingLoss.cpp",
                                                         "CosineEmbeddingLossNorm2d",
                                                         build_params));

    auto reduce_size        = problem.GetInput1Desc().GetLengths()[1];
    auto output_numel       = problem.GetInput1Desc().GetLengths()[0] * 3;
    auto reqd_work_item_cnt = get_reqd_work_item_cnt(context, LOCAL_SIZE_REDUCED_SUM);
    if(is_parallelism(reqd_work_item_cnt, output_numel, reduce_size))
    {
        auto parallelism_size = get_parallelism_size(reqd_work_item_cnt, output_numel, reduce_size);
        result.construction_params.push_back(make_hip_kernel({LOCAL_SIZE_REDUCED_SUM},
                                                             {parallelism_size * output_numel},
                                                             "MIOpenSum.cpp",
                                                             "SumParallelFwdContiguous",
                                                             build_params));
    }
    result.construction_params.push_back(make_hip_kernel({LOCAL_SIZE_REDUCED_SUM},
                                                         {output_numel},
                                                         "MIOpenSum.cpp",
                                                         "SumFwdContiguous",
                                                         build_params));
}

inline void RunNormKernels(const std::vector<Kernel>& kernels,
                           const Handle& handle_,
                           const AnyInvokeParams& raw_params,
                           int& kernelCnt,
                           Data_t& work_a,
                           Data_t& work_b)
{
    auto params = raw_params.CastTo<miopen::cosineembeddingloss::FwdInvokeParams>();

    {
        auto I1_tv  = get_inner_expanded_tv_2d(deref(params.input1Desc));
        auto I2_tv  = get_inner_expanded_tv_2d(deref(params.input2Desc));
        auto kernel = handle_.Run(kernels[kernelCnt++]);

        kernel(params.input1, params.input2, work_a, I1_tv, I2_tv);
    }

    auto reduce_size        = params.input1Desc->GetLengths()[1];
    auto output_numel       = params.input1Desc->GetLengths()[0] * 3;
    auto reqd_work_item_cnt = get_reqd_work_item_cnt(handle_, LOCAL_SIZE_REDUCED_SUM);

    if(is_parallelism(reqd_work_item_cnt, output_numel, reduce_size))
    {
        auto parallelism_size = get_parallelism_size(reqd_work_item_cnt, output_numel, reduce_size);
        auto parallel_kernel  = handle_.Run(kernels[kernelCnt++]);
        parallel_kernel(work_a,
                        work_b,
                        (uint64_t)output_numel,
                        (uint64_t)reduce_size,
                        (uint64_t)parallelism_size,
                        (uint64_t)1,
                        false);

        auto kernel = handle_.Run(kernels[kernelCnt++]);
        kernel(
            work_b, work_a, (uint64_t)output_numel, (uint64_t)parallelism_size, (uint64_t)1, false);
    }
    else
    {
        auto kernel = handle_.Run(kernels[kernelCnt++]);
        kernel(work_a, work_b, (uint64_t)output_numel, (uint64_t)reduce_size, (uint64_t)1, false);
        std::swap(work_a, work_b);
    }
}

bool CosineEmbeddingLossUnreducedForward2d::IsApplicable(
    const ExecutionContext&,
    const miopen::cosineembeddingloss::FwdUnreducedProblemDescription& problem) const
{
    if(!problem.IsValidLength())
        return false;
    return true;
}

ConvSolution CosineEmbeddingLossUnreducedForward2d::GetSolution(
    const ExecutionContext& context,
    const miopen::cosineembeddingloss::FwdUnreducedProblemDescription& problem) const
{
    auto result       = ConvSolution{miopenStatusSuccess};
    auto input_dtype  = miopen::GetDataType(problem.GetInput1Desc().GetType());
    auto output_dtype = miopen::GetDataType(problem.GetOutputDesc().GetType());

    auto dtype     = problem.GetOutputDesc().GetType();
    size_t N_total = problem.GetNtotal();

    const auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
        {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
        {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
        {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
        {"OUTPUT_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
        {"D_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
    };

    ConstructNormParamsKernels(context, problem, result, build_params);

    result.construction_params.push_back(make_hip_kernel({LOCAL_SIZE_UNREDUCED_FWD},
                                                         {N_total},
                                                         "MIOpenCosineEmbeddingLoss.cpp",
                                                         "CosineEmbeddingLossUnreducedForward2d",
                                                         build_params));

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) params =
                raw_params.CastTo<miopen::cosineembeddingloss::FwdInvokeParams>();

            float elapsed = 0.0f;
            int kernelCnt = 0;

            auto work_a = params.workspace;
            auto work_b =
                reinterpret_cast<Data_t>(reinterpret_cast<char*>(params.workspace) +
                                         params.input1Desc->GetElementSize() *
                                             get_data_size(params.outputDesc->GetType()) * 3);

            HipEventPtr start;
            HipEventPtr stop;

            if(handle_.IsProfilingEnabled())
            {
                start = miopen::make_hip_event();
                stop  = miopen::make_hip_event();
                hipEventRecord(start.get(), handle_.GetStream());
            }

            RunNormKernels(kernels, handle_, raw_params, kernelCnt, work_a, work_b);

            auto target_tv = get_inner_expanded_tv_1d(deref(params.targetDesc));
            auto output_tv = get_inner_expanded_tv_1d(deref(params.outputDesc));

            auto kernel = handle_.Run(kernels[kernelCnt++]);
            kernel(work_a, params.target, params.output, params.margin, target_tv, output_tv);

            if(handle_.IsProfilingEnabled())
            {
                hipEventRecord(stop.get(), handle_.GetStream());
                hipEventSynchronize(stop.get());
                hipEventElapsedTime(&elapsed, start.get(), stop.get());
                handle_.ResetKernelTime();
                handle_.AccumKernelTime(elapsed);
            };
        };
    };

    return result;
}

std::size_t CosineEmbeddingLossUnreducedForward2d::GetWorkspaceSize(
    const ExecutionContext& context,
    const miopen::cosineembeddingloss::FwdUnreducedProblemDescription& problem) const
{
    std::size_t size = problem.GetInput1Desc().GetElementSize() *
                       get_data_size(problem.GetOutputDesc().GetType()) * 3;

    auto reduce_size        = problem.GetInput1Desc().GetLengths()[1];
    auto output_numel       = problem.GetInput1Desc().GetLengths()[0] * 3;
    auto reqd_work_item_cnt = get_reqd_work_item_cnt(context, LOCAL_SIZE_REDUCED_SUM);
    if(is_parallelism(reqd_work_item_cnt, output_numel, reduce_size))
    {
        auto parallelism_size = get_parallelism_size(reqd_work_item_cnt, output_numel, reduce_size);
        size += parallelism_size * output_numel * get_data_size(problem.GetOutputDesc().GetType());
    }
    else
    {
        size += output_numel * get_data_size(problem.GetOutputDesc().GetType());
    }
    return size;
}

} // namespace cosineembeddingloss

} // namespace solver

} // namespace miopen
