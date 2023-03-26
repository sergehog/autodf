/*
 * This file is part of the AutoDf distribution (https://github.com/sergehog/autodf)
 * Copyright (c) 2023 Sergey Smirnov / Seregium Oy.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "../autodf.h"

using namespace autodf;

//! Basic Const checks
template <unsigned ID>
constexpr int test_Const(const Variable<ID> x)
{
    constexpr auto five = Const{5.0};
    static_assert(5.F == five.eval());
    static_assert(5.F == five.eval({}));
    static_assert(5.F == five.eval<1>({0.0}));
    static_assert(5.F == five.eval<1>({1000.0}));
    static_assert(0.F == five.gradient({1.F}).at(0));
    static_assert(0.F == five.gradient<1>({1.F}).at(0));

    static_assert(10.F == (five + x).eval({5.F}));
    // static_assert(1.F == (five-x).eval({4.F}));
    static_assert(10.F == (five * x).eval({2.F}));
    // static_assert(2.5F == (five/x).eval({2.F}));

    static_assert(10.F == (five + 5.F).eval());
    static_assert(1.F == (five - 4.F).eval());
    static_assert(10.F == (five * 2.F).eval());
    static_assert(2.5F == (five / 2.F).eval());

    static_assert(11.F == (6.F + five).eval());
    static_assert(6.F == (11.F - five).eval());
    static_assert(15.F == (3.F * five).eval());
    static_assert(3.F == (15.F / five).eval());

    return 0;
}

//! Basic Variable checks
template <unsigned ID0>
constexpr Variable<1> test_Variable(const Variable<ID0> x)
{
    constexpr Variable<1> y;
    static_assert(0 == Variable<0>::MAXID);
    static_assert(1.F == x.eval({1.F}));
    static_assert(1.F == x.gradient({0.F})[0]);
    static_assert(1 == Variable<1>::MAXID);
    static_assert(11111.F == y.eval({0, 11111.F}));
    static_assert(1.F == y.gradient({111.F, 0.123F})[Variable<1>::MAXID]);
    static_assert(0.F == y.gradient({0.123F, 123.F})[0]);
    static_assert(0.F == y.gradient({0.123F, 123.F}).at(0));

    return y;
}

//! Basic Sum Checks
template <unsigned ID0, unsigned ID1>
constexpr int test_Sum(const Variable<ID0> x, const Variable<ID1> y)
{
    constexpr auto five = Const{5.0};

    constexpr auto x_plus_y = Sum{x, y};  // manual lvalue Sum
    static_assert(5.F == x_plus_y.eval({2.F, 3.F}));
    static_assert(8.F == Sum{x, y}.eval({3.F, 5.F}));  // manual rvalue Sum
    static_assert(10.F == (x + y).eval({3.F, 7.F}));   // automatic Sum
    // manual rvalue Mul
    static_assert(15.F == Mul{x, y}.eval({3.F, 5.F}));
    static_assert(25.F == (x * y).eval({5.F, 5.F}));
    // Variable + Const lvalue
    constexpr auto x_plus_five = Sum{x, five};
    static_assert((2.f + 5.F) == x_plus_five.eval({2.F}));
    // Const + Variable rvalue
    static_assert((2.f + 5.F) == (five + x).eval({2.F}));
    return 0;
}

//! eval() checks
template <unsigned ID0, unsigned ID1>
constexpr int test_eval(const Variable<ID0> x, const Variable<ID1> y)
{
    static_assert((2.f + 5.F) == Sum{x, Const{5.F}}.eval({2.F}));
    constexpr auto x_plus_5 = x + 5.F;
    static_assert((2.f + 5.F) == x_plus_5.eval({2.F}));
    constexpr auto x5 = x * 5.F;
    static_assert((2.f * 5.F) == x5.eval({2.F}));
    constexpr auto _5x = 5.F * x;
    static_assert(x5.eval({2.F}) == _5x.eval({2.F}));
    static_assert(2.F == (x + y).eval({1.F, 1.F}));
    static_assert(2.F == (x * y + y * x).eval({1.F, 1.F}));
    static_assert(2.F == (x * x + y * y).eval({1.F, 1.F}));
    static_assert(8.F == (x * x + y * y).eval({2.F, 2.F}));
    return 0;
}

//! gradient() checks
template <unsigned ID0, unsigned ID1>
constexpr int test_gradient(const Variable<ID0> x, const Variable<ID1> y)
{
    constexpr auto g = Sum{x, y}.gradient({1.F, 1.F});
    static_assert(1.0 == g[0]);
    static_assert(1.0 == g[1]);

    constexpr auto gMul = Mul{x, y}.gradient({1.F, 1.F});
    static_assert(1.0 == gMul[0]);
    static_assert(1.0 == gMul[1]);

    constexpr auto f = (x + (-1.F)) * (x + 1.F) + (y + (-1.F)) * (y + 1.F);
    constexpr auto fg10 = f.gradient({1.F, 0.F});
    static_assert(fg10[0] == 2.0);
    static_assert(fg10[1] == 0.0);

    constexpr auto fg_10 = f.gradient({-1.F, 0.F});
    static_assert(fg_10[0] == -2.0);
    static_assert(fg_10[1] == 0.0);

    constexpr auto fg01 = f.gradient({0.F, 1.F});
    static_assert(fg01[0] == 0.0);
    static_assert(fg01[1] == 2.0);

    constexpr auto fg_01 = f.gradient({0.F, -1.F});
    static_assert(fg_01[0] == 0.0);
    static_assert(fg_01[1] == -2.0);

    constexpr auto fg00 = f.gradient({0.F, 0.F});
    static_assert(fg01[0] == 0.0);
    static_assert(fg01[0] == 0.0);
    return 0;
}



int main()
{
    // initialization of variable
    constexpr Variable<0> x;

    // test Const structure
    test_Const(x);  // test also passing variable as argument

    constexpr auto y = test_Variable(x);  // test return variable

    // test Sum structure
    test_Sum(x, y);

    // test eval() function
    test_eval(x, y);

    // test gradient() function
    test_gradient(x, y);

    return 0;
}
