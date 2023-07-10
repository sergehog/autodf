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

#ifndef AUTODF_H
#define AUTODF_H

#include <array>

namespace autodf
{
// Forward declaration for Mul;
template <typename T1, typename T2>
struct Mul;

// Forward declaration for Sum;
template <typename T1, typename T2>
struct Sum;

// Forward declaration for Sub;
template <typename T1, typename T2>
struct Sub;

// Forward declaration for Variable;
template <unsigned ID>
struct Variable;

///////////////////////////////////////////////////////////////////////////////////////////////
//! Represents Constant
struct Const
{
    static constexpr unsigned MAXID = 0;
    explicit constexpr Const(const double v) : value(v) {}
    const double value;

    template <unsigned AMNT = MAXID + 1>
    [[nodiscard]] constexpr double eval([[maybe_unused]] const std::array<double, AMNT>& unused = {}) const
    {
        return value;
    }

    template <unsigned forID, unsigned AMNT = MAXID + 1>
    [[nodiscard]] constexpr double gradient([[maybe_unused]] const std::array<double, AMNT>& unused) const
    {
        return 0.0;
    }

    //! Sum where Const is left argument : Res = Const + Other
    template <typename T2>
    constexpr Sum<const Const, const T2> operator+(T2 other) const
    {
        return Sum<const Const, const T2>{*this, other};
    }

    template <typename T2>
    constexpr Sub<const Const, const T2> operator-(T2 other) const
    {
        return Sub<const Const, const T2>{*this, other};
    }

    //! Mul where Const is left argument : Res = Const * Other
    template <typename T2>
    constexpr Mul<const Const, const T2> operator*(T2 other) const
    {
        return Mul<const Const, const T2>{*this, other};
    }

    // operations with other Const
    constexpr Const operator+(const Const other) const { return Const{value + other.value}; }

    constexpr Const operator-(const Const other) const { return Const{value - other.value}; }

    constexpr Const operator*(const Const other) const { return Const{value * other.value}; }

    constexpr Const operator/(const Const other) const { return Const{value / other.value}; }

    // operations with scalar
    constexpr Const operator+(const double other) const { return Const{value + other}; }

    constexpr Const operator-(const double other) const { return Const{value - other}; }

    constexpr Const operator*(const double other) const { return Const{value * other}; }

    constexpr Const operator/(const double other) const { return Const{value / other}; }
};

constexpr Const operator+(const double a, const Const& b)
{
    return Const{a + b.value};
}

constexpr Const operator-(const double a, const Const& b)
{
    return Const{a - b.value};
}

constexpr Const operator*(const double a, const Const& b)
{
    return Const{a * b.value};
}

constexpr Const operator/(const double a, const Const& b)
{
    return Const{a / b.value};
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//! Represents a variable, template parameter ID defines variable uniqueness
template <unsigned ID = 0>
struct Variable
{
    static constexpr unsigned MAXID = ID;

    template <unsigned AMNT = MAXID + 1>
    [[nodiscard]] constexpr double eval(const std::array<double, AMNT>& input) const
    {
        return input[ID];
    }

    template <unsigned forID, unsigned AMNT = MAXID + 1>
    [[nodiscard]] constexpr double gradient([[maybe_unused]] const std::array<double, AMNT>& input) const
    {
        return forID == ID ? 1.0 : 0.0;
    }

    constexpr Sum<const Variable<ID>, const Const> operator+(const double value) const
    {
        return Sum<const Variable<ID>, const Const>{*this, Const{value}};
    }

    constexpr Sub<const Variable<ID>, const Const> operator-(const double value) const
    {
        return Sub<const Variable<ID>, const Const>{*this, Const{value}};
    }

    constexpr Mul<const Variable<ID>, const Const> operator*(const double value) const
    {
        return Mul<const Variable<ID>, const Const>{*this, Const{value}};
    }

    template <typename T2>
    constexpr Sum<const Variable<ID>, const T2> operator+(T2 value) const
    {
        return Sum<const Variable<ID>, const T2>{*this, value};
    }
    template <typename T2>
    constexpr Sub<const Variable<ID>, const T2> operator-(T2 value) const
    {
        return Sub<const Variable<ID>, const T2>{*this, value};
    }

    template <typename T2>
    constexpr Mul<const Variable<ID>, const T2> operator*(T2 value) const
    {
        return Mul<const Variable<ID>, const T2>{*this, value};
    }

    constexpr Sub<const Const, const Variable<ID>> operator-() const
    {
        return Sub<const Const, const Variable<ID>>{Const{0}, this};
    }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! Defines Multiplication
template <typename T1, typename T2>
struct Mul
{
    static constexpr unsigned MAXID = T1::MAXID > T2::MAXID ? T1::MAXID : T2::MAXID;
    constexpr Mul(const T1 ai, const T2 bi) : a(ai), b(bi) {}
    const T1 a;
    const T2 b;

    template <unsigned AMNT = MAXID + 1>
    [[nodiscard]] constexpr double eval(const std::array<double, AMNT>& input) const
    {
        return a.template eval<AMNT>(input) * b.template eval<AMNT>(input);
    }

    template <unsigned forID, unsigned AMNT = MAXID + 1>
    [[nodiscard]] constexpr double gradient(const std::array<double, AMNT>& input) const
    {
        return a.template gradient<forID, AMNT>(input) * b.template eval<AMNT>(input) +
               b.template gradient<forID, AMNT>(input) * a.template eval<AMNT>(input);
    }

    constexpr Sum<const Mul<T1, T2>, const Const> operator+(double value) const
    {
        return Sum<const Mul<T1, T2>, const Const>{*this, Const{value}};
    }

    constexpr Sub<const Mul<T1, T2>, const Const> operator-(double value) const
    {
        return Sub<const Mul<T1, T2>, const Const>{*this, Const{value}};
    }

    constexpr Mul<const Mul<T1, T2>, const Const> operator*(double value) const
    {
        return Mul<const Mul<T1, T2>, const Const>{*this, Const{value}};
    }

    template <typename T3>
    constexpr Sum<const Mul<T1, T2>, const T3> operator+(T3 value) const
    {
        return Sum<const Mul<T1, T2>, const T3>{*this, value};
    }

    template <typename T3>
    constexpr Sub<const Mul<T1, T2>, const T3> operator-(T3 value) const
    {
        return Sub<const Mul<T1, T2>, const T3>{*this, value};
    }

    template <typename T3>
    constexpr Mul<const Mul<T1, T2>, const T3> operator*(T3 value) const
    {
        return Mul<const Mul<T1, T2>, const T3>{*this, value};
    }

    constexpr Sub<const Const, const Mul<T1, T2>> operator-() const
    {
        return Sub<const Const, const Mul<T1, T2>>{Const{0}, *this};
    }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! Defines Summation
template <typename T1, typename T2>
struct Sum
{
    static constexpr unsigned MAXID = T1::MAXID > T2::MAXID ? T1::MAXID : T2::MAXID;

    constexpr Sum(T1 ai, T2 bi) : a(ai), b(bi) {}

    const T1 a;
    const T2 b;

    template <unsigned AMNT = MAXID + 1>
    [[nodiscard]] constexpr double eval(const std::array<double, AMNT>& input) const
    {
        return a.template eval<AMNT>(input) + b.template eval<AMNT>(input);
    }

    template <unsigned forID, unsigned AMNT = MAXID + 1>
    [[nodiscard]] constexpr double gradient(const std::array<double, AMNT>& input) const
    {
        return a.template gradient<forID, AMNT>(input) + b.template gradient<forID, AMNT>(input);
    }

    constexpr Sum<const Sum<T1, T2>, const Const> operator+(double value) const
    {
        return Sum<const Sum<T1, T2>, const Const>{*this, Const{value}};
    }

    constexpr Sub<const Sum<T1, T2>, const Const> operator-(double value) const
    {
        return Sub<const Sum<T1, T2>, const Const>{*this, Const{value}};
    }

    constexpr Mul<const Sum<T1, T2>, const Const> operator*(double value) const
    {
        return Mul<const Sum<T1, T2>, const Const>{*this, Const{value}};
    }

    template <typename T3>
    constexpr Sum<const Sum<T1, T2>, const T3> operator+(T3 value) const
    {
        return Sum<const Sum<T1, T2>, const T3>{*this, value};
    }

    template <typename T3>
    constexpr Sub<const Sum<T1, T2>, const T3> operator-(T3 value) const
    {
        return Sub<const Sum<T1, T2>, const T3>{*this, value};
    }

    template <typename T3>
    constexpr Mul<const Sum<T1, T2>, const T3> operator*(T3 value) const
    {
        return Mul<const Sum<T1, T2>, const T3>{*this, value};
    }

    constexpr Sub<const Const, const Sum<T1, T2>> operator-() const
    {
        return Sub<const Const, const Sum<T1, T2>>{Const{0}, *this};
    }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! Defines Subtraction
template <typename T1, typename T2>
struct Sub
{
    static constexpr unsigned MAXID = T1::MAXID > T2::MAXID ? T1::MAXID : T2::MAXID;

    constexpr Sub(const T1 ai, const T2 bi) : a(ai), b(bi) {}

    const T1 a;
    const T2 b;

    template <unsigned AMNT = MAXID + 1>
    [[nodiscard]] constexpr double eval(const std::array<double, AMNT>& input) const
    {
        return a.template eval<AMNT>(input) - b.template eval<AMNT>(input);
    }

    template <unsigned forID, unsigned AMNT = MAXID + 1>
    [[nodiscard]] constexpr double gradient(const std::array<double, AMNT>& input) const
    {
        return a.template gradient<forID, AMNT>(input) - b.template gradient<forID, AMNT>(input);
    }

    constexpr Sum<const Sub<T1, T2>, const Const> operator+(double value) const
    {
        return Sum<const Sub<T1, T2>, const Const>{*this, Const{value}};
    }

    constexpr Sub<const Sub<T1, T2>, const Const> operator-(double value) const
    {
        return Sub<const Sub<T1, T2>, const Const>{*this, Const{value}};
    }

    constexpr Mul<const Sub<T1, T2>, const Const> operator*(double value) const
    {
        return Mul<const Sub<T1, T2>, const Const>{*this, Const{value}};
    }

    template <typename T3>
    constexpr Sum<const Sub<T1, T2>, const T3> operator+(T3 value) const
    {
        return Sum<const Sub<T1, T2>, const T3>{*this, value};
    }

    template <typename T3>
    constexpr Sub<const Sub<T1, T2>, const T3> operator-(T3 value) const
    {
        return Sub<const Sub<T1, T2>, const T3>{*this, value};
    }

    template <typename T3>
    constexpr Mul<const Sub<T1, T2>, const T3> operator*(T3 value) const
    {
        return Mul<const Sub<T1, T2>, const T3>{*this, value};
    }

    constexpr Sub<const Const, const Sub<T1, T2>> operator-() const
    {
        return Sub<const Const, const Sub<T1, T2>>{Const{0}, *this};
    }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! Generic operators
template <unsigned ID>
constexpr Sum<const Const, const Variable<ID>> operator+(const double a, const Variable<ID> b)
{
    return Sum<const Const, const Variable<ID>>{Const{a}, b};
}

template <unsigned ID>
constexpr Sub<const Const, const Variable<ID>> operator-(const double a, const Variable<ID> b)
{
    return Sub<const Const, const Variable<ID>>{Const{a}, b};
}

template <unsigned ID>
constexpr Mul<const Const, const Variable<ID>> operator*(const double a, const Variable<ID> b)
{
    return Mul<const Const, const Variable<ID>>{Const{a}, b};
}

template <typename T2>
constexpr Sum<const Const, const T2> operator+(const double a, const T2 b)
{
    return Sum<const Const, const T2>{Const{a}, b};
}

template <typename T2>
constexpr Sub<const Const, const T2> operator-(const double a, const T2 b)
{
    return Sub<const Const, const T2>{Const{a}, b};
}

template <typename T2>
constexpr Mul<const Const, const T2> operator*(const double a, T2 b)
{
    return Mul<const Const, const T2>{Const{a}, b};
}
}  // namespace autodf

#endif  // AUTODF_H
