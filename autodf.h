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
template<typename T1, typename T2> struct Mul;

// Forward declaration for Sum;
template<typename T1, typename T2> struct Sum;

// Forward declaration for Variable;
template<unsigned ID> struct Variable;


///////////////////////////////////////////////////////////////////////////////////////////////
//! Represents Constant
struct Const
{
    static constexpr unsigned MAXID = 0;
    constexpr Const(const float v) : value(v) {}
    const float value;

    template<unsigned AMNT=MAXID+1>
    [[nodiscard]] constexpr float eval(const std::array<float, AMNT>&) const
    {
        return value;
    }

    template<unsigned AMNT=MAXID+1>
    [[nodiscard]] constexpr std::array<float, AMNT> gradient(const std::array<float, AMNT>&)  const
    {
        return std::array<float, AMNT>{};
    }

    //  template<unsigned AMNT=MAXID+1>
    //  [[nodiscard]] constexpr std::array<float, AMNT> gradient(const std::array<float, AMNT>&)  const
    //  {
    //    return std::array<float, AMNT>{};
    //  }

    template<typename T2> constexpr Sum<const Const,const T2> operator+(T2 other) const;
    template<typename T2> constexpr Mul<const Const,const T2> operator*(T2 other) const;
};

template<typename T2> constexpr Sum<const Const,const T2> Const::operator+(T2 other) const
{
    return Sum<const Const,const T2> {*this, other};
}

template<typename T2> constexpr Mul<const Const,const T2> Const::operator*(T2 other) const
{
    return Mul<const Const,const T2> {*this, other};
}


////////////////////////////////////////////////////////////////////////////////////////////////////
//! Represents a variable
template<unsigned ID=0>
struct Variable
{
    static constexpr unsigned MAXID = ID;

    template<unsigned AMNT=MAXID+1>
    [[nodiscard]] constexpr float eval(const std::array<float, AMNT>& input) const
    {
        return input[ID];
    }

    template<unsigned AMNT=MAXID+1>
    [[nodiscard]] constexpr std::array<float, AMNT> gradient(const std::array<float, AMNT>& input) const
    {
        std::array<float, AMNT> out {};
        out[ID] = 1.F;
        return out;
    }

    constexpr Sum<const Variable<ID>,const Const> operator+(float value) const
    {
        return Sum<const Variable<ID>,const Const> {*this, Const{value}};
    }
    constexpr Mul<const Variable<ID>,const Const> operator*(float value) const
    {
        return Mul<const Variable<ID>,const Const> {*this, Const{value}};
    }

    template<typename T2> constexpr Sum<const Variable<ID>,const T2> operator+(T2 value) const
    {
        return Sum<const Variable<ID>,const T2> {*this, value};
    }
    template<typename T2> constexpr Mul<const Variable<ID>,const T2> operator*(T2 value) const
    {
        return Mul<const Variable<ID>,const T2> {*this, value};
    }
};


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! Defines Multiplication
template<typename T1, typename T2>
struct Mul
{
    static constexpr unsigned MAXID = T1::MAXID > T2::MAXID ? T1::MAXID : T2::MAXID;
    constexpr Mul(T1 ai, T2 bi) : a(ai), b(bi) {}
    T1 a;
    T2 b;

    template<unsigned AMNT=MAXID+1>
    [[nodiscard]] constexpr float eval(const std::array<float, AMNT>& input) const
    {
        return a.template eval<AMNT>(input) * b.template eval<AMNT>(input);
    }

    template<unsigned AMNT=MAXID+1>
    [[nodiscard]] constexpr std::array<float, AMNT> gradient(const std::array<float, AMNT>& input) const
    {
        const float av = a.template eval<AMNT>(input);
        const float bv = b.template eval<AMNT>(input);
        const std::array<float, AMNT> ga = a.template gradient<AMNT>(input);
        const std::array<float, AMNT> gb = b.template gradient<AMNT>(input);
        std::array<float, AMNT> out {};
        for(unsigned id = 0; id < AMNT; id ++)
        {
            out[id] = ga[id]*bv + gb[id]*av;
        }
        return out;
    }

    constexpr Sum<const Mul<T1, T2>,const Const> operator+(float value) const
    {
        return Sum<const Mul<T1, T2>,const Const> {*this, Const{value}};
    }

    constexpr Mul<const Mul<T1, T2>,const Const> operator*(float value) const
    {
        return Mul<const Mul<T1, T2>,const Const> {*this, Const{value}};
    }

    template<typename T3> constexpr Sum<const Mul<T1, T2>,const T3> operator+(T3 value) const
    {
        return Sum<const Mul<T1, T2>,const T3> {*this, value};
    }

    template<typename T3> constexpr Mul<const Mul<T1, T2>, const T3> operator*(T3 value) const
    {
        return Mul<const Mul<T1, T2>,const T3> {*this, value};
    }

};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! Defines Summation
template<typename T1, typename T2>
struct Sum
{
    static constexpr unsigned MAXID = T1::MAXID > T2::MAXID ? T1::MAXID : T2::MAXID;

    constexpr Sum(T1 ai, T2 bi) : a(ai), b(bi) {}

    T1 a;
    T2 b;

    template<unsigned AMNT=MAXID+1>
    [[nodiscard]] constexpr float eval(const std::array<float, AMNT>& input) const
    {
        return a.template eval<AMNT>(input) + b.template eval<AMNT>(input);
    }

    template<unsigned AMNT=MAXID+1>
    [[nodiscard]] constexpr std::array<float, AMNT> gradient(const std::array<float, AMNT>& input) const
    {
        const auto ga = a.template gradient<AMNT>(input);
        const auto gb = b.template gradient<AMNT>(input);
        std::array<float, AMNT> out {};
        for(unsigned id = 0; id < AMNT; id ++)
        {
            out[id] = ga[id] + gb[id];
        }
        return out;
    }

    constexpr Sum<const Sum<T1, T2>,const Const> operator+(float value) const
    {
        return Sum<const Sum<T1, T2>,const Const> {*this, Const{value}};
    }

    constexpr Mul<const Sum<T1, T2>,const Const> operator*(float value) const
    {
        return Mul<const Sum<T1, T2>,const Const> {*this, Const{value}};
    }

    template<typename T3> constexpr Sum<const Sum<T1, T2>,const T3> operator+(T3 value) const
    {
        return Sum<const Sum<T1, T2>,const T3> {*this, value};
    }

    template<typename T3> constexpr Mul<const Sum<T1, T2>, const T3> operator*(T3 value) const
    {
        return Mul<const Sum<T1, T2>,const T3> {*this, value};
    }
};


//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! Generic operators
template<unsigned ID>
constexpr Sum<const Const, const Variable<ID>> operator+ (const float a, Variable<ID>& b)
{
    return Sum<const Const, const Variable<ID>>{Const{a}, b};
}

template<unsigned ID>
constexpr Mul<const Const, const Variable<ID>> operator* (const float a, Variable<ID>& b)
{
    return Mul<const Const, const Variable<ID>>{Const{a}, b};
}

template<typename T2>
constexpr Sum<const Const, const T2> operator+ (const float a, T2& b)
{
    return Sum<const Const, const T2>{Const{a}, b};
}

template<typename T2>
constexpr Mul<const Const, const T2> operator* (const float a, T2& b)
{
    return Mul<const Const, const T2>{Const{a}, b};
}
}

#endif  // AUTODF_H
