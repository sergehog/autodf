/*
 * This file is part of the AutoDf distribution (https://github.com/sergehog/autodf)
 * Copyright (c) 2023-2024 Sergey Smirnov / Seregium Oy
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
constexpr int testConst(const Variable<ID> x)
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
constexpr Variable<1> testVariable(const Variable<ID0> x)
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
constexpr int testSum(const Variable<ID0> x, const Variable<ID1> y)
{
    constexpr auto five = Const{5.0};
    constexpr auto x_plus_y = Sum{x, y};  // manual lvalue Sum
    static_assert(5. == x_plus_y.eval({2., 3.}));
    static_assert(1. == x_plus_y.template gradient<0U, 2U>({5, 5}));
    static_assert(8. == Sum{x, y}.eval({3., 5.}));  // manual rvalue Sum
    static_assert(10. == (x + y).eval({3., 7.}));   // automatic Sum
    // manual rvalue
    static_assert(20. == (5 + Mul{x, y}).eval({3., 5.}));
    static_assert(10. == (x + five).template eval<2>({5., 5.}));
    // Variable + Const lvalue
    constexpr auto x_plus_five = Sum{x, five};
    static_assert((2. + 5.) == x_plus_five.eval({2.}));
    // Const + Variable rvalue
    static_assert((2. + 5.) == (5. + x).eval({2.}));
    return 0;
}

//! Basic Sum Checks
template <unsigned ID0, unsigned ID1>
constexpr int testSub(const Variable<ID0> x, const Variable<ID1> y)
{
    constexpr auto five = Const{5.0};

    constexpr auto x_minus_y = Sub{x, y};  // manual lvalue Sum
    static_assert(1. == x_minus_y.eval({3., 2.}));
    static_assert(-2. == Sub{x, y}.eval({3., 5.}));  // manual rvalue Sum
    static_assert(-4. == (x - y).eval({3., 7.}));    // automatic Sum
    // manual rvalue Mul
    static_assert(15. == Mul{x, y}.eval({3., 5.}));
    static_assert(0. == (x - five).template eval<2>({5., 5.}));
    // Variable + Const lvalue
    constexpr auto x_minus_five = Sub{x, five};
    static_assert((2. - 5.) == x_minus_five.eval({2.}));
    // Const + Variable rvalue
    static_assert((2. - 5.) == (x - five).eval({2.}));
    return 0;
}

//! Basic Sum Checks
template <unsigned ID0, unsigned ID1>
constexpr int testMul(const Variable<ID0> x, const Variable<ID1> y)
{
    constexpr auto five = Const{5.0};
    constexpr auto x_times_y = Mul{x, y};
    static_assert(6. == x_times_y.eval({3., 2.F}));
    static_assert(15. == Mul{x, y}.eval({3., 5.F}));
    static_assert(21. == (x * y).eval({3., 7.}));
    // manual rvalue
    static_assert(15. == Mul{x, y}.eval({3., 5.}));
    static_assert(25. == (x * five).template eval<2>({5., 5.}));
    // Variable + Const lvalue
    constexpr auto x_times_five = Mul{x, five};
    static_assert((2. * 5.) == x_times_five.eval({2.}));
    // Const + Variable rvalue
    static_assert((2. * 5.) == (x * five).eval({2.}));
    return 0;
}

template <unsigned ID0>
constexpr int testSin(const Variable<ID0> x)
{
    constexpr auto s = sin(x);
    int result = (s.template eval({0}) == 0.0) ? 0 : 1;
    result = (s.template gradient<0, 1>({0}) == 1.0) ? result : 1;

    constexpr auto s2 = sin(x - 2.0);
    result = std::abs(s2.template eval({0}) - std::sin(-2.0)) < 1e-8 ? result : 2;
    result = std::abs(s2.template gradient<0, 1>({0}) - std::cos(-2.0)) < 1e-8 ? result : 3;
    return result;
}

//! eval() checks
template <unsigned ID0, unsigned ID1>
constexpr int testEval(const Variable<ID0> x, const Variable<ID1> y)
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
constexpr int testGradient(const Variable<ID0> x, const Variable<ID1> y)
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

int testRuntimeExpr()
{
    constexpr autodf::Variable<0> c01;
    constexpr autodf::Variable<1> c02;
    constexpr autodf::Variable<2> c03;
    constexpr autodf::Variable<3> c12;
    constexpr autodf::Variable<4> c13;
    constexpr autodf::Variable<5> c23;
    constexpr autodf::Variable<6> vel;
    constexpr autodf::Variable<7> acc;
    constexpr autodf::Variable<8> steer;

    constexpr autodf::Variable<9> dt;
    constexpr autodf::Variable<10> L;

    constexpr auto C01 =
        ((-(0.08333333333333333 * c01 * dt * dt * sin(steer) * sin(steer))) -
         0.08333333333333333 * L * c12 * dt * dt * cos(steer) * sin(steer)) *
            vel * vel +
        ((0.1666666666666666 * c03 * c23 + 0.1666666666666666 * c01 * c12 - c02 / 2.0) * dt * sin(steer) +
         (0.1666666666666666 * L * c13 * c13 + 0.1666666666666666 * L * c12 * c12 - L / 2.0) * dt * cos(steer)) *
            vel +
        c01;

    constexpr auto C02 =
        (-(0.08333333333333333 * c02 * dt * dt * sin(steer) * sin(steer) * vel * vel)) +
        (((-(0.1666666666666666 * c03 * c13)) + 0.1666666666666666 * c02 * c12 + c01 / 2.0) * dt * sin(steer) +
         (0.1666666666666666 * L * c13 * c23 + L / 2.0 * c12) * dt * cos(steer)) *
            vel +
        c02;  //  # e0 ^ e2
    constexpr auto C03 = (-(0.08333333333333333 * L * c23 * dt * dt * cos(steer) * sin(steer) * vel * vel)) +
                         ((0.3333333333333333 * c02 * c13 - 0.3333333333333333 * c01 * c23) * dt * sin(steer) +
                          (L / 2.0 * c13 - 0.1666666666666666 * L * c12 * c23) * dt * cos(steer)) *
                             vel +
                         c03;  //  # e0 ^ e3
    constexpr auto C12 =
        ((-(0.1666666666666666 * c23 * c23)) - 0.1666666666666666 * c13 * c13 + 0.5) * dt * sin(steer) * vel +
        c12;  //#e1 ^ e2
    constexpr auto C13 = (-(0.08333333333333333 * c13 * dt * dt * sin(steer) * sin(steer) * vel * vel)) +
                         (0.1666666666666666 * c12 * c13 - c23 / 2.0) * dt * sin(steer) * vel + c13;  //  # e1 ^ e3
    constexpr auto C23 = (-(0.08333333333333333 * c23 * dt * dt * sin(steer) * sin(steer) * vel * vel)) +
                         (0.1666666666666666 * c12 * c23 + c13 / 2.0) * dt * sin(steer) * vel + c23;  // #e2 ^ e3

    constexpr std::array<double, 11> values{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    auto grad = C01.template gradient<0, 11>(values);

    int result = (C01.template eval<11>(values) >= 0.F) ? 0 : 1;
    result = (C02.template eval<11>(values) >= 0.F) ? result : 2;
    result = (C03.template eval<11>(values) >= 0.F) ? result : 3;
    result = (C12.template eval<11>(values) >= 0.F) ? result : 4;
    result = (C13.template eval<11>(values) >= 0.F) ? result : 5;
    result = (C23.template eval<11>(values) >= 0.F) ? result : 6;
    return result;
}

int main()
{
    // initialization of variable
    constexpr Variable<0> x;

    // tests Const structure, also passing constexpr Variable as argument
    testConst(x);

    // tests and returns constexpr Variable
    constexpr auto y = testVariable(x);

    testSum(x, y);

    testSub(x, y);

    testMul(x, y);

    // tests eval() function
    testEval(x, y);

    // tests gradient() function
    testGradient(x, y);

    if (const auto res = testSin(x) > 0)
    {
        return res;
    }

    return testRuntimeExpr();
}
