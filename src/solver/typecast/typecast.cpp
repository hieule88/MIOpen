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

#include <miopen/datatype.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/tensor_view_utils.hpp>
#include <miopen/typecast.hpp>
#include <miopen/typecast/invoke_params.hpp>
#include <miopen/typecast/solvers.hpp>

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace typecast {

namespace {
inline std::string ConvertTypeName(std::string mi_type)
{
    if(mi_type == "bfloat16")
        return "ushort";
    else if(mi_type == "int64")
        return "int64_t";
    else if(mi_type == "int8_t")
        return "__hip_internal::int8_t";
    else
        return mi_type;
}
} // unnamed namespace

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
        {"IS_I_TYPE_BFP16", static_cast<int>(input_dtype == miopenBFloat16)},
        {"IS_I_TYPE_FP32", static_cast<int>(input_dtype == miopenFloat)},
        {"IS_I_TYPE_FP64", static_cast<int>(input_dtype == miopenDouble)},
        {"IS_O_TYPE_BFP16", static_cast<int>(output_dtype == miopenBFloat16)},
        {"I_TYPE", ConvertTypeName(mi_input_dtype)},
        {"O_TYPE", ConvertTypeName(mi_output_dtype)},
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
