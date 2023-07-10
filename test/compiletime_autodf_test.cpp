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
    static_assert(5.0 == five.eval());
    static_assert(5.0 == five.eval({}));
    static_assert(5.0 == five.eval<1>({0.0}));
    static_assert(5.0 == five.eval<1>({1000.0}));
    static_assert(0.0 == five.gradient<0>({1.F}));
    static_assert(0.0 == five.gradient<1>({1.F}));

    static_assert(10.0 == (five + x).eval({5.0}));
    static_assert(1.0 == (five - x).eval({4.0}));
    static_assert(10.0 == (five * x).eval({2.0}));
    // static_assert(2.5 == (five/x).eval({2.F}));

    static_assert(10.0 == (five + 5.0).eval<1>({5.F}));
    static_assert(1.0 == (five - 4.0).eval<1>({}));
    static_assert(10.0 == (five * 2.0).eval<1>({}));
    static_assert(2.50 == (five / 2.0).eval<1>({}));

    static_assert(11.0 == (6.0 + five).eval());
    static_assert(6.0 == (11.0 - five).eval());
    static_assert(15.0 == (3.0 * five).eval());
    static_assert(3.0 == (15.0 / five).eval());

    return 0;
}

//! Basic Variable checks
template <unsigned ID0>
constexpr Variable<1> test_Variable(const Variable<ID0> x)
{
    constexpr Variable<1> y;
    static_assert(0 == Variable<0>::MAXID);
    static_assert(1.0 == x.eval({1.F}));
    static_assert(1.0 == x.template gradient<0>({0.0}));
    static_assert(1.0 == Variable<1>::MAXID);
    static_assert(11111.0 == y.eval({0, 11111.0}));
    static_assert(1.0 == y.template gradient<1>({111.0, 0.123}));
    static_assert(0.0 == y.template gradient<0>({0.123, 123.0}));
    static_assert(0.0 == y.template gradient<0>({0.123, 123.0}));

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
    static_assert((2.0 + 5.0) == Sum{x, Const{5.0}}.eval({2.0}));
    constexpr auto x_plus_5 = x + 5.0;
    static_assert((2.0 + 5.0) == x_plus_5.eval({2.0}));
    constexpr auto x5 = x * 5.0;
    static_assert((2.0 * 5.0) == x5.eval({2.0}));
    constexpr auto _5x = 5.0 * x;
    static_assert(x5.eval({2.0}) == _5x.eval({2.0}));
    static_assert(2.0 == (x + y).eval({1.0, 1.0}));
    static_assert(2.0 == (x * y + y * x).eval({1.0, 1.0}));
    static_assert(2.0 == (x * x + y * y).eval({1.0, 1.0}));
    static_assert(8.0 == (x * x + y * y).eval({2.0, 2.0}));
    return 0;
}

//! gradient() checks
template <unsigned ID0, unsigned ID1>
constexpr int test_gradient(const Variable<ID0> x, const Variable<ID1> y)
{
    constexpr auto g0 = Sum{x, y}.template gradient<0>({1.0, 2.0});
    constexpr auto g1 = Sum{x, y}.template gradient<1>({1.0, 2.0});
    static_assert(1.0 == g0);
    static_assert(1.0 == g1);

    constexpr auto gMul0 = Mul{x, y}.template gradient<0>({1.F, 1.F});
    constexpr auto gMul1 = Mul{x, y}.template gradient<1>({1.F, 1.F});
    static_assert(1.0 == gMul0);
    static_assert(1.0 == gMul1);

    constexpr auto f = (x + (-1.0)) * (x + 1.0) + (y + (-1.0)) * (y + 1.0);

    static_assert(f.template gradient<0>({1.0, 0.0}) == 2.0);
    static_assert(f.template gradient<1>({1.0, 0.0}) == 0.0);

    static_assert(f.template gradient<0>({-1.0, 0.0}) == -2.0);
    static_assert(f.template gradient<1>({-1.0, 0.0}) == 0.0);

    static_assert(f.template gradient<0>({0.0, 1.0}) == 0.0);
    static_assert(f.template gradient<1>({0.0, 1.0}) == 2.0);

    static_assert(f.template gradient<0>({0.0, -1.0}) == 0.0);
    static_assert(f.template gradient<1>({0.0, -1.0}) == -2.0);

    static_assert(f.template gradient<0>({0.0, 0.0}) == 0.0);
    static_assert(f.template gradient<1>({0.0, 0.0}) == 0.0);
    return 0;
}

int main()
{
    // initialization of variable
    constexpr Variable<0> x;

    // tests Const structure, also passing constexpr Variable as argument
    test_Const(x);

    // tests and returns constexpr Variable
    constexpr auto y = test_Variable(x);

    // tests Sum structure
    test_Sum(x, y);

    // tests eval() function
    test_eval(x, y);

    // tests gradient() function
    test_gradient(x, y);

    return 0;
}
