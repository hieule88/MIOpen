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
#include "cpu_typecast.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/typecast.hpp>
#include <miopen/miopen.h>
#include <sys/types.h>

template <class T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
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

struct TypeCastTestCase
{
    std::vector<size_t> input_dim;
    uint64_t bits_to_truncate;

    friend std::ostream& operator<<(std::ostream& os, const TypeCastTestCase& tc)
    {
        return os << " input_dim:" << tc.input_dim << " bits_to_truncate:" << tc.bits_to_truncate;
    }
};

inline std::vector<TypeCastTestCase> TypeCastTestConfigs()
{
    return {
        {{10, 10, 10, 10}, 0},
        {{10, 100, 10, 10}, 1},
        {{10, 10, 10, 10, 10}, 0},
        {{10, 100, 10, 10, 10}, 1},
    };
}

// FORWARD TEST
template <typename TI, typename TO>
struct TypeCastTestFwd : public ::testing::TestWithParam<TypeCastTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle   = get_handle();
        typecast_config = GetParam();
        in_dim          = typecast_config.input_dim;

        auto gen_input_value = [](auto...) {
            return prng::gen_A_to_B<TI>(static_cast<TI>(-10.0f), static_cast<TI>(10.0f));
        };
        input = tensor<TI>{in_dim}.generate(gen_input_value);

        output = tensor<TO>{in_dim};
        std::fill(output.begin(), output.end(), 0.0f);

        ref_output = tensor<TO>{in_dim};
        std::fill(ref_output.begin(), ref_output.end(), 0.0f);

        input_dev  = handle.Write(input.data);
        output_dev = handle.Write(output.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();
        miopenStatus_t status;

        cpu_typecast<TI, TO>(input, ref_output, bits_to_truncate);

        status = miopen::typecast::TypeCast(
            handle, input.desc, input_dev.get(), output.desc, output_dev.get(), bits_to_truncate);

        ASSERT_EQ(status, miopenStatusSuccess);
        output.data = handle.Read<TO>(output_dev, output.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<TO>::epsilon();

        auto error = miopen::rms_range(ref_output, output);

        for(int i = 0; i < 10; i++)
        {
            std::cout << "CPU ref_output[" << i << "] = " << ref_output[i] << " GPU output[" << i
                      << "] = " << output[i] << std::endl;
        }

        ASSERT_EQ(miopen::range_distance(ref_output), miopen::range_distance(output));

        EXPECT_LE(error, threshold * 10) << "Error forward Output beyond 10xthreshold : " << error
                                         << " Tolerance: " << threshold * 10;
    }
    TypeCastTestCase typecast_config;

    std::vector<size_t> in_dim;

    tensor<TI> input;
    tensor<TO> output;
    tensor<TO> ref_output;

    uint64_t bits_to_truncate;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr output_dev;
};
