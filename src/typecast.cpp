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
#include <miopen/kernel_cache.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/tensor.hpp>
#include <miopen/typecast/invoke_params.hpp>
#include <miopen/typecast/solvers.hpp>
#include <miopen/find_solution.hpp>

namespace miopen {

namespace typecast {

miopenStatus_t TypeCast(Handle& handle,
                        const TensorDescriptor& inputDesc,
                        ConstData_t input,
                        const TensorDescriptor& outputDesc,
                        Data_t output,
                        const uint64_t bits_to_truncate)
{
    const auto problem = typecast::ProblemDescription{inputDesc, outputDesc, bits_to_truncate};
    const auto invoke_params = [&]() {
        auto tmp             = typecast::InvokeParams{};
        tmp.inputDesc        = &inputDesc;
        tmp.input            = input;
        tmp.outputDesc       = &outputDesc;
        tmp.output           = output;
        tmp.bits_to_truncate = bits_to_truncate;

        return tmp;
    }();
    const auto algo    = AlgorithmName{"TypeCast"};
    const auto solvers = solver::SolverContainer<solver::typecast::TypeCast>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);
    return miopenStatusSuccess;
}

} // namespace typecast

} // namespace miopen
