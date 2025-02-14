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
#include "registry_driver_maker.hpp"
#include "typecast_driver.hpp"

static Driver* makeDriver(const std::string& base_arg)
{
    // float32 to other types
    if(base_arg == "typecast_float32half")
        return new TypeCastDriver<float, half>();
    if(base_arg == "typecast_float32bfloat16")
        return new TypeCastDriver<float, bfloat16>();
    if(base_arg == "typecast_float32int8")
        return new TypeCastDriver<float, int8_t>();
    if(base_arg == "typecast_float32int")
        return new TypeCastDriver<float, int>();
    if(base_arg == "typecast_float32double")
        return new TypeCastDriver<float, double>();
    // if(base_arg == "typecast_float32float8")
    // return new TypeCastDriver<float, float8>();
    // if(base_arg == "typecast_float32bfloat8")
    // return new TypeCastDriver<float, bfloat8>();
    if(base_arg == "typecast_float32int64")
        return new TypeCastDriver<float, int64_t>();

    // half to other types
    if(base_arg == "typecast_bfloat16")
        return new TypeCastDriver<half, bfloat16>();
    if(base_arg == "typecast_int8")
        return new TypeCastDriver<half, int8_t>();
    if(base_arg == "typecast_int")
        return new TypeCastDriver<half, int>();
    if(base_arg == "typecast_double")
        return new TypeCastDriver<half, double>();
    // if(base_arg == "typecast_float8")
    // return new TypeCastDriver<half, float8>();
    // if(base_arg == "typecast_bfloat8")
    // return new TypeCastDriver<half, bfloat8>();
    if(base_arg == "typecast_int64")
        return new TypeCastDriver<half, int64_t>();

    // bfloat16 to other types
    if(base_arg == "typecast_bfloat16int8")
        return new TypeCastDriver<bfloat16, int8_t>();
    if(base_arg == "typecast_bfloat16int")
        return new TypeCastDriver<bfloat16, int>();
    if(base_arg == "typecast_bfloat16double")
        return new TypeCastDriver<bfloat16, double>();
    // if(base_arg == "typecast_bfloat16float8")
    // return new TypeCastDriver<bfloat16, float8>();
    // if(base_arg == "typecast_bfloat16bfloat8")
    // return new TypeCastDriver<bfloat16, bfloat8>();
    if(base_arg == "typecast_bfloat16int64")
        return new TypeCastDriver<bfloat16, int64_t>();

    // int8 to other types
    if(base_arg == "typecast_int8int")
        return new TypeCastDriver<int8_t, int>();
    if(base_arg == "typecast_int8double")
        return new TypeCastDriver<int8_t, double>();
    // if(base_arg == "typecast_int8float8")
    // return new TypeCastDriver<int8_t, float8>();
    // if(base_arg == "typecast_int8bfloat8")
    // return new TypeCastDriver<int8_t, bfloat8>();
    if(base_arg == "typecast_int8int64")
        return new TypeCastDriver<int8_t, int64_t>();

    // int to other types
    if(base_arg == "typecast_intdouble")
        return new TypeCastDriver<int, double>();
    // if(base_arg == "typecast_intfloat8")
    // return new TypeCastDriver<int, float8>();
    // if(base_arg == "typecast_intbfloat8")
    // return new TypeCastDriver<int, bfloat8>();
    if(base_arg == "typecast_intint64")
        return new TypeCastDriver<int, int64_t>();

    // double to other types
    // if(base_arg == "typecast_doublefloat8")
    // return new TypeCastDriver<double, float8>();
    // if(base_arg == "typecast_doublebfloat8")
    // return new TypeCastDriver<double, bfloat8>();
    if(base_arg == "typecast_doubleint64")
        return new TypeCastDriver<double, int64_t>();

    // // float8 to other types
    // if(base_arg == "typecast_float8bfloat8")
    // return new TypeCastDriver<float8, bfloat8>();
    // if(base_arg == "typecast_float8int64")
    // return new TypeCastDriver<float8, int64_t>();

    // // bfloat8 to other types
    // if(base_arg == "typecast_bfloat8int64")
    // return new TypeCastDriver<bfloat8, int64_t>();

    // other type to float32
    if(base_arg == "typecast_halffloat32")
        return new TypeCastDriver<half, float>();
    if(base_arg == "typecast_bfloat16float32")
        return new TypeCastDriver<bfloat16, float>();
    if(base_arg == "typecast_int8float32")
        return new TypeCastDriver<int8_t, float>();
    if(base_arg == "typecast_intfloat32")
        return new TypeCastDriver<int, float>();
    if(base_arg == "typecast_doublefloat32")
        return new TypeCastDriver<double, float>();
    // if(base_arg == "typecast_float8float32")
    // return new TypeCastDriver<float8, float>();
    // if(base_arg == "typecast_bfloat8float32")
    // return new TypeCastDriver<bfloat8, float>();
    if(base_arg == "typecast_int64float32")
        return new TypeCastDriver<int64_t, float>();

    // other type to half
    if(base_arg == "typecast_bfloat16half")
        return new TypeCastDriver<bfloat16, half>();
    if(base_arg == "typecast_int8half")
        return new TypeCastDriver<int8_t, half>();
    if(base_arg == "typecast_inthalf")
        return new TypeCastDriver<int, half>();
    if(base_arg == "typecast_doublehalf")
        return new TypeCastDriver<double, half>();
    // if(base_arg == "typecast_float8half")
    // return new TypeCastDriver<float8, half>();
    // if(base_arg == "typecast_bfloat8half")
    // return new TypeCastDriver<bfloat8, half>();
    if(base_arg == "typecast_int64half")
        return new TypeCastDriver<int64_t, half>();

    // other type to bfloat16
    if(base_arg == "typecast_int8bfloat16")
        return new TypeCastDriver<int8_t, bfloat16>();
    if(base_arg == "typecast_intbfloat16")
        return new TypeCastDriver<int, bfloat16>();
    if(base_arg == "typecast_doublebfloat16")
        return new TypeCastDriver<double, bfloat16>();
    // if(base_arg == "typecast_float8bfloat16")
    // return new TypeCastDriver<float8, bfloat16>();
    // if(base_arg == "typecast_bfloat8bfloat16")
    // return new TypeCastDriver<bfloat8, bfloat16>();
    if(base_arg == "typecast_int64bfloat16")
        return new TypeCastDriver<int64_t, bfloat16>();

    // other type to int8
    if(base_arg == "typecast_intint8")
        return new TypeCastDriver<int, int8_t>();
    if(base_arg == "typecast_doubleint8")
        return new TypeCastDriver<double, int8_t>();
    // if(base_arg == "typecast_float8int8")
    // return new TypeCastDriver<float8, int8_t>();
    // if(base_arg == "typecast_bfloat8int8")
    // return new TypeCastDriver<bfloat8, int8_t>();
    if(base_arg == "typecast_int64int8")
        return new TypeCastDriver<int64_t, int8_t>();

    // other type to int32
    if(base_arg == "typecast_doubleint")
        return new TypeCastDriver<double, int>();
    // if(base_arg == "typecast_float8int")
    // return new TypeCastDriver<float8, int>();
    // if(base_arg == "typecast_bfloat8int")
    // return new TypeCastDriver<bfloat8, int>();
    if(base_arg == "typecast_int64int")
        return new TypeCastDriver<int64_t, int>();

    // other type to double
    // if(base_arg == "typecast_float8double")
    // return new TypeCastDriver<float8, double>();
    // if(base_arg == "typecast_bfloat8double")
    // return new TypeCastDriver<bfloat8, double>();
    if(base_arg == "typecast_int64double")
        return new TypeCastDriver<int64_t, double>();

    // // other type to float8
    // if(base_arg == "typecast_bfloat8float8")
    // return new TypeCastDriver<bfloat8, float8>();
    // if(base_arg == "typecast_int64float8")
    // return new TypeCastDriver<int64_t, float8>();

    // // other type to bfloat8
    // if(base_arg == "typecast_int64bfloat8")
    // return new TypeCastDriver<int64_t, bfloat8>();

    return nullptr;
}

REGISTER_DRIVER_MAKER(makeDriver);
