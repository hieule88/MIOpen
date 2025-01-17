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
#include <miopen/typecast/invoke_params.hpp>
#include <miopen/typecast/solvers.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/datatype.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/tensor_view_utils.hpp>

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace typecast {

bool TypeCast::IsApplicable(const ExecutionContext&,
                            const miopen::typecast::ProblemDescription& problem) const
{
    return true;
}

ConvSolution TypeCast::GetSolution(const ExecutionContext& context,
                                   const miopen::typecast::ProblemDescription& problem) const
{
    std::ignore          = context;
    auto mi_input_dtype  = miopen::GetDataType(problem.GetInputDesc().GetType());
    auto mi_output_dtype = miopen::GetDataType(problem.GetOutputDesc().GetType());
    auto input_dtype     = problem.GetInputDesc().GetType();
    auto output_dtype    = problem.GetOutputDesc().GetType();
    auto numel           = problem.GetOutputDesc().GetElementSize();

    auto result       = ConvSolution{miopenStatusSuccess};
    auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16",
         static_cast<int>(output_dtype == miopenHalf || input_dtype == miopenHalf)},
        {"MIOPEN_USE_FP32",
         static_cast<int>(output_dtype == miopenFloat || input_dtype == miopenFloat)},
        {"MIOPEN_USE_FP64",
         static_cast<int>(output_dtype == miopenDouble || input_dtype == miopenDouble)},
        {"MIOPEN_USE_BFP16",
         static_cast<int>(output_dtype == miopenBFloat16 || input_dtype == miopenBFloat16)},
        {"I_TYPE", mi_input_dtype == "bfloat16" ? "ushort" : mi_input_dtype},
        {"O_TYPE", mi_output_dtype == "int64" ? "size_t" : mi_output_dtype},
    };

    if(!problem.IsAllContiguous())
    {
        result.construction_params.push_back(
            make_hip_kernel({LOCAL_SIZE}, {numel}, "MIOpenTypeCast.cpp", "TypeCast", build_params));

        result.invoker_factory = [numel](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel = handle_.Run(kernels[0]);
                decltype(auto) params = raw_params.CastTo<miopen::typecast::InvokeParams>();
                auto input_tv         = get_inner_expanded_tv<5>(deref(params.inputDesc));
                auto output_tv        = get_inner_expanded_tv<5>(deref(params.outputDesc));

                kernel(params.input,
                       params.output,
                       params.bits_to_truncate,
                       numel,
                       input_tv,
                       output_tv);
            };
        };
    }
    else
    {
        result.construction_params.push_back(make_hip_kernel(
            {LOCAL_SIZE}, {numel}, "MIOpenTypeCast.cpp", "TypeCastContiguous", build_params));

        result.invoker_factory = [numel](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel = handle_.Run(kernels[0]);
                decltype(auto) params = raw_params.CastTo<miopen::typecast::InvokeParams>();

                kernel(params.input, params.output, params.bits_to_truncate, numel);
            };
        };
    }

    return result;
};

} // namespace typecast

} // namespace solver

} // namespace miopen
