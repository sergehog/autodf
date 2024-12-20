/*
 * This file is part of the AutoDf distribution (https://github.com/sergehog/autodf)
 * Copyright (c) 2023-2024 Sergey Smirnov / Seregium Oy.
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
#include <cmath>

namespace autodf
{
// Forward declaration for Mul;
template <typename T1, typename T2>
struct Mul;

// Forward declaration for Div;
template <typename T1, typename T2>
struct Div;

// Forward declaration for Sum;
template <typename T1, typename T2>
struct Sum;

// Forward declaration for Sub;
template <typename T1, typename T2>
struct Sub;

// Forward declaration for Variable;
template <unsigned ID>
struct Variable;

#define CONST_OPS(TypeName)                                                           \
    constexpr Sum<const TypeName, const Const> operator+(const double value_in) const \
    {                                                                                 \
        return Sum<const TypeName, const Const>{*this, Const{value_in}};              \
    }                                                                                 \
    constexpr Sub<const TypeName, const Const> operator-(const double value_in) const \
    {                                                                                 \
        return Sub<const TypeName, const Const>{*this, Const{value_in}};              \
    }                                                                                 \
    constexpr Mul<const TypeName, const Const> operator*(const double value_in) const \
    {                                                                                 \
        return Mul<const TypeName, const Const>{*this, Const{value_in}};              \
    }                                                                                 \
    constexpr Mul<const TypeName, const Const> operator/(const double value_in) const \
    {                                                                                 \
        return Mul<const TypeName, const Const>{*this, Const{1.0 / value_in}};        \
    }

#define GENERIC_OPS(TypeName)                                               \
    template <typename TX>                                                  \
    constexpr Sum<const TypeName, const TX> operator+(const TX other) const \
    {                                                                       \
        return Sum<const TypeName, const TX>{*this, other};                 \
    }                                                                       \
    template <typename TX>                                                  \
    constexpr Sub<const TypeName, const TX> operator-(const TX other) const \
    {                                                                       \
        return Sub<const TypeName, const TX>{*this, other};                 \
    }                                                                       \
    template <typename TX>                                                  \
    constexpr Mul<const TypeName, const TX> operator*(const TX other) const \
    {                                                                       \
        return Mul<const TypeName, const TX>{*this, other};                 \
    }                                                                       \
    template <typename TX>                                                  \
    constexpr Div<const TypeName, const TX> operator/(const TX other) const \
    {                                                                       \
        return Div<const TypeName, const TX>{*this, other};                 \
    }

///////////////////////////////////////////////////////////////////////////////////////////////
//! Constant
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

    GENERIC_OPS(Const)
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

    CONST_OPS(Variable<ID>)
    GENERIC_OPS(Variable<ID>)
};
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

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! Multiplication
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

    using TypeName = Mul<T1, T2>;
    CONST_OPS(TypeName)
    GENERIC_OPS(TypeName)
};
//! Left operand with normal const
template <typename T2>
constexpr Mul<const Const, const T2> operator*(const double a, T2 b)
{
    return Mul<const Const, const T2>{Const{a}, b};
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! Division
template <typename T1, typename T2>
struct Div
{
    static constexpr unsigned MAXID = T1::MAXID > T2::MAXID ? T1::MAXID : T2::MAXID;
    constexpr Div(const T1 ai, const T2 bi) : a(ai), b(bi) {}
    const T1 a;
    const T2 b;

    template <unsigned AMNT = MAXID + 1>
    [[nodiscard]] constexpr double eval(const std::array<double, AMNT>& input) const
    {
        return a.template eval<AMNT>(input) / b.template eval<AMNT>(input);
    }

    template <unsigned forID, unsigned AMNT = MAXID + 1>
    [[nodiscard]] constexpr double gradient(const std::array<double, AMNT>& input) const
    {
        return (a.template gradient<forID, AMNT>(input) * b.template eval<AMNT>(input) -
                b.template gradient<forID, AMNT>(input) * a.template eval<AMNT>(input)) /
               (b.template eval<AMNT>(input) * b.template eval<AMNT>(input));
    }

    using TypeName = Div<T1, T2>;
    CONST_OPS(TypeName)
    GENERIC_OPS(TypeName)
};
//! Left operand with normal const
template <typename T2>
constexpr Div<const Const, const T2> operator/(const double a, T2 b)
{
    return Div<const Const, const T2>{Const{a}, b};
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! Summation
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

    using TypeName = Sum<T1, T2>;
    CONST_OPS(TypeName)
    GENERIC_OPS(TypeName)
};

//! Left operand with normal const
template <typename T2>
constexpr Sum<const Const, const T2> operator+(const double a, const T2 b)
{
    return Sum<const Const, const T2>{Const{a}, b};
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! Subtraction
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

    using TypeName = Sub<T1, T2>;
    CONST_OPS(TypeName)
    GENERIC_OPS(TypeName)
};
//! Left operand with normal const
template <typename T1>
constexpr Sub<const Const, const T1> operator-(const double a, const T1 b)
{
    return Sub<const Const, const T1>{Const{a}, b};
}

//! unary minus
template <typename T1>
constexpr Sub<const Const, const T1> operator-(const T1 a)
{
    return Sub<const Const, const T1>{Const{0.0}, a};
}

///////////////////////////////////////////////////////////////////////////////////////////////
//! Sin() function
template <typename T1>
struct Sin
{
    static constexpr unsigned MAXID = T1::MAXID;

    explicit constexpr Sin(const T1 v) : value(v) {}

    const T1 value;

    template <unsigned AMNT = MAXID + 1>
    [[nodiscard]] constexpr double eval(const std::array<double, AMNT>& input = {}) const
    {
        return std::sin(value.template eval<AMNT>(input));
    }

    template <unsigned forID, unsigned AMNT = MAXID + 1>
    [[nodiscard]] constexpr double gradient(const std::array<double, AMNT>& input) const
    {
        return value.template gradient<forID, AMNT>(input) * std::cos(value.template eval<AMNT>(input));
    }

    using TypeName = Sin<T1>;
    CONST_OPS(TypeName)
    GENERIC_OPS(TypeName)
};
template <typename T1>
constexpr Sin<const T1> sin(const T1 a)
{
    return Sin<const T1>{a};
}

///////////////////////////////////////////////////////////////////////////////////////////////
//! Asin() function
template <typename T1>
struct Asin
{
    static constexpr unsigned MAXID = T1::MAXID;

    explicit constexpr Asin(const T1 v) : value(v) {}

    const T1 value;

    template <unsigned AMNT = MAXID + 1>
    [[nodiscard]] constexpr double eval(const std::array<double, AMNT>& input = {}) const
    {
        return std::asin(value.template eval<AMNT>(input));
    }

    template <unsigned forID, unsigned AMNT = MAXID + 1>
    [[nodiscard]] constexpr double gradient(const std::array<double, AMNT>& input) const
    {
        return value.template gradient<forID, AMNT>(input) /
               std::sqrt(1. - std::pow(value.template eval<AMNT>(input), 2.));
    }

    using TypeName = Asin<T1>;
    CONST_OPS(TypeName)
    GENERIC_OPS(TypeName)
};
template <typename T1>
constexpr Asin<const T1> asin(const T1 a)
{
    return Asin<const T1>{a};
}

///////////////////////////////////////////////////////////////////////////////////////////////
//! Cos() function
template <typename T1>
struct Cos
{
    static constexpr unsigned MAXID = T1::MAXID;

    explicit constexpr Cos(const T1 v) : value(v) {}

    const T1 value;

    template <unsigned AMNT = MAXID + 1>
    [[nodiscard]] constexpr double eval(const std::array<double, AMNT>& input = {}) const
    {
        return std::cos(value.template eval<AMNT>(input));
    }

    template <unsigned forID, unsigned AMNT = MAXID + 1>
    [[nodiscard]] constexpr double gradient(const std::array<double, AMNT>& input) const
    {
        return -value.template gradient<forID, AMNT>(input) * std::sin(value.template eval<AMNT>(input));
    }

    using TypeName = Cos<T1>;
    CONST_OPS(TypeName)
    GENERIC_OPS(TypeName)
};
template <typename T1>
constexpr Cos<const T1> cos(const T1 a)
{
    return Cos<const T1>{a};
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! Atan2
template <typename T1, typename T2>
struct Atan2
{
    static constexpr unsigned MAXID = T1::MAXID > T2::MAXID ? T1::MAXID : T2::MAXID;

    constexpr Atan2(T1 yi, T2 xi) : a(yi), b(xi) {}

    const T1 a;
    const T2 b;

    template <unsigned AMNT = MAXID + 1>
    [[nodiscard]] constexpr double eval(const std::array<double, AMNT>& input) const
    {
        return std::atan2(a.template eval<AMNT>(input), b.template eval<AMNT>(input));
    }

    template <unsigned forID, unsigned AMNT = MAXID + 1>
    [[nodiscard]] constexpr double gradient(const std::array<double, AMNT>& input) const
    {
        const auto db_dt = b.template gradient<forID, AMNT>(input);
        const auto da_dt = a.template gradient<forID, AMNT>(input);
        const auto b_t = b.template eval<AMNT>(input);
        const auto a_t = a.template eval<AMNT>(input);
        const auto norm2 = a_t * a_t + b_t * b_t;

        const auto datan2_a_t = b_t / norm2;
        const auto datan2_b_t = -a_t / norm2;
        return datan2_a_t * da_dt + datan2_b_t * db_dt;
    }

    using TypeName = Atan2<T1, T2>;
    CONST_OPS(TypeName)
    GENERIC_OPS(TypeName)
};

template <typename T1, typename T2>
constexpr Atan2<const T1, const T2> atan2(const T1 a, const T2 b)
{
    return Atan2<const T1, const T2>{a, b};
}

///////////////////////////////////////////////////////////////////////////////////////////////
//! Sqrt() function
template <typename T1>
struct Sqrt
{
    static constexpr unsigned MAXID = T1::MAXID;

    explicit constexpr Sqrt(const T1 v) : value(v) {}

    const T1 value;

    template <unsigned AMNT = MAXID + 1>
    [[nodiscard]] constexpr double eval(const std::array<double, AMNT>& input = {}) const
    {
        return std::sqrt(value.template eval<AMNT>(input));
    }

    template <unsigned forID, unsigned AMNT = MAXID + 1>
    [[nodiscard]] constexpr double gradient(const std::array<double, AMNT>& input) const
    {
        return (0.5 / std::sqrt(value.template eval<AMNT>(input))) * value.template gradient<forID, AMNT>(input);
    }

    using TypeName = Sqrt<T1>;
    CONST_OPS(TypeName)
    GENERIC_OPS(TypeName)
};
template <typename T1>
constexpr Sqrt<const T1> sqrt(const T1 a)
{
    return Sqrt<const T1>{a};
}

///////////////////////////////////////////////////////////////////////////////////////////////
//! IfPositive(COND, A, B) function
template <typename T1, typename T2, typename T3>
struct IfPositive
{
    static constexpr unsigned MAXID = T1::MAXID > T2::MAXID ? ((T1::MAXID > T3::MAXID) ? T1::MAXID : T3::MAXID)
                                                            : ((T2::MAXID > T3::MAXID) ? T2::MAXID : T3::MAXID);

    const T1 condition;
    const T2 valueIfTrue;
    const T3 valueIfFalse;

    explicit constexpr IfPositive(const T1 eq, const T2 ifTrue, const T3 ifFalse)
        : condition(eq), valueIfTrue(ifTrue), valueIfFalse(ifFalse)
    {
    }

    template <unsigned AMNT = MAXID + 1>
    [[nodiscard]] constexpr double eval(const std::array<double, AMNT>& input = {}) const
    {
        if (condition.template eval<AMNT>(input) > 0.0)
        {
            return valueIfTrue.template eval<AMNT>(input);
        }
        else
        {
            return valueIfFalse.template eval<AMNT>(input);
        }
    }

    template <unsigned forID, unsigned AMNT = MAXID + 1>
    [[nodiscard]] constexpr double gradient(const std::array<double, AMNT>& input) const
    {
        if (condition.template eval<AMNT>(input) > 0.0)
        {
            return valueIfTrue.template gradient<forID, AMNT>(input);
        }
        else
        {
            return valueIfFalse.template gradient<forID, AMNT>(input);
        }
    }
    using TypeName = IfPositive<T1, T2, T3>;
    CONST_OPS(TypeName)
    GENERIC_OPS(TypeName)
};
template <typename T1, typename T2, typename T3>
constexpr IfPositive<const T1, const T2, const T3> ifPositive(const T1 condition, const T2 ifTrue, const T3 ifFalse)
{
    return IfPositive<const T1, const T2, const T3>{condition, ifTrue, ifFalse};
}

template <typename T1, typename T2>
constexpr IfPositive<const T1, const T2, Const> ifPositive(const T1 condition, const T2 ifTrue, const double ifFalse)
{
    return IfPositive<const T1, const T2, Const>{condition, ifTrue, Const{ifFalse}};
}

template <typename T1, typename T3>
constexpr IfPositive<const T1, Const, const T3> ifPositive(const T1 condition, const double ifTrue, const T3 ifFalse)
{
    return IfPositive<const T1, Const, const T3>{condition, Const{ifTrue}, ifFalse};
}

constexpr double ifPositive(const double condition, const double ifTrue, const double ifFalse)
{
    return (condition > 0.0) ? ifTrue : ifFalse;
}

}  // namespace autodf

#endif  // AUTODF_H
