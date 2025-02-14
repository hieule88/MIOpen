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
#include "typecast.hpp"
#include "miopen/bfloat16.hpp"
using float16 = half_float::half;

using GPU_TypeCast_fwd_int64bfloat16 = TypeCastTestFwd<int64_t, bfloat16>;
using GPU_TypeCast_fwd_floatbfloat16 = TypeCastTestFwd<float, bfloat16>;
using GPU_TypeCast_fwd_int64float    = TypeCastTestFwd<int64_t, float>;
using GPU_TypeCast_fwd_bfloat16float = TypeCastTestFwd<bfloat16, float>;
using GPU_TypeCast_fwd_uint8float    = TypeCastTestFwd<uint8_t, float>;
using GPU_TypeCast_fwd_int8float     = TypeCastTestFwd<int8_t, float>;
using GPU_TypeCast_fwd_boolfloat     = TypeCastTestFwd<int8_t, float>;
using GPU_TypeCast_fwd_uint8int64    = TypeCastTestFwd<uint8_t, int64_t>;
using GPU_TypeCast_fwd_float32int64  = TypeCastTestFwd<float, int64_t>;
using GPU_TypeCast_fwd_boolint64     = TypeCastTestFwd<int8_t, int64_t>;
using GPU_TypeCast_fwd_float32uint8  = TypeCastTestFwd<float, uint8_t>;
using GPU_TypeCast_fwd_uint8bool     = TypeCastTestFwd<uint8_t, int8_t>;
using GPU_TypeCast_fwd_bfloat16bool  = TypeCastTestFwd<bfloat16, int8_t>;
using GPU_TypeCast_fwd_float32bool   = TypeCastTestFwd<float, int8_t>;

TEST_P(GPU_TypeCast_fwd_int64bfloat16, Test)
{
    RunTest();
    Verify();
};

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_TypeCast_fwd_int64bfloat16,
                         testing::ValuesIn(TypeCastTestConfigs()));

TEST_P(GPU_TypeCast_fwd_floatbfloat16, Test)
{
    RunTest();
    Verify();
};

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_TypeCast_fwd_floatbfloat16,
                         testing::ValuesIn(TypeCastTestConfigs()));

TEST_P(GPU_TypeCast_fwd_int64float, Test)
{
    RunTest();
    Verify();
};

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_TypeCast_fwd_int64float,
                         testing::ValuesIn(TypeCastTestConfigs()));

TEST_P(GPU_TypeCast_fwd_bfloat16float, Test)
{
    RunTest();
    Verify();
};

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_TypeCast_fwd_bfloat16float,
                         testing::ValuesIn(TypeCastTestConfigs()));

TEST_P(GPU_TypeCast_fwd_uint8float, Test)
{
    RunTest();
    Verify();
};

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_TypeCast_fwd_uint8float,
                         testing::ValuesIn(TypeCastTestConfigs()));

TEST_P(GPU_TypeCast_fwd_int8float, Test)
{
    RunTest();
    Verify();
};

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_TypeCast_fwd_int8float,
                         testing::ValuesIn(TypeCastTestConfigs()));

TEST_P(GPU_TypeCast_fwd_boolfloat, Test)
{
    RunTest();
    Verify();
};

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_TypeCast_fwd_boolfloat,
                         testing::ValuesIn(TypeCastTestConfigs()));

TEST_P(GPU_TypeCast_fwd_uint8int64, Test)
{
    RunTest();
    Verify();
};

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_TypeCast_fwd_uint8int64,
                         testing::ValuesIn(TypeCastTestConfigs()));

TEST_P(GPU_TypeCast_fwd_float32int64, Test)
{
    RunTest();
    Verify();
};

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_TypeCast_fwd_float32int64,
                         testing::ValuesIn(TypeCastTestConfigs()));

TEST_P(GPU_TypeCast_fwd_boolint64, Test)
{
    RunTest();
    Verify();
};

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_TypeCast_fwd_boolint64,
                         testing::ValuesIn(TypeCastTestConfigs()));

TEST_P(GPU_TypeCast_fwd_float32uint8, Test)
{
    RunTest();
    Verify();
};

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_TypeCast_fwd_float32uint8,
                         testing::ValuesIn(TypeCastTestConfigs()));

TEST_P(GPU_TypeCast_fwd_uint8bool, Test)
{
    RunTest();
    Verify();
};

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_TypeCast_fwd_uint8bool,
                         testing::ValuesIn(TypeCastTestConfigs()));

TEST_P(GPU_TypeCast_fwd_bfloat16bool, Test)
{
    RunTest();
    Verify();
};

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_TypeCast_fwd_bfloat16bool,
                         testing::ValuesIn(TypeCastTestConfigs()));

TEST_P(GPU_TypeCast_fwd_float32bool, Test)
{
    RunTest();
    Verify();
};

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_TypeCast_fwd_float32bool,
                         testing::ValuesIn(TypeCastTestConfigs()));
