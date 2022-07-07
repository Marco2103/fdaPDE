#ifndef __SCALAR_FIELD_EXPRESSIONS_H__
#define __SCALAR_FIELD_EXPRESSIONS_H__

#include "../Symbols.h"

#include <functional>
#include <type_traits>

namespace fdaPDE{
namespace core{

// ScalarField (and its derived classes as consequece) support expression template based arithmetic.
// The goal is to allow to write expressions of scalar fields which are lazily evaluated only when a point evaluation is requested.

// macro to define an arithmetic operator between scalar fields.
#define DEF_FIELD_EXPR_OPERATOR(OPERATOR, FUNCTOR)			\
  template <typename E1, typename E2>					\
  FieldBinOp<E1, E2, FUNCTOR >						\
  OPERATOR(const FieldExpr<E1>& op1, const FieldExpr<E2>& op2) {	\
    return FieldBinOp<E1, E2, FUNCTOR >					\
      {op1.get(), op2.get(), FUNCTOR()};				\
  }									\
  									\
  template <typename E>							\
  FieldBinOp<E, FieldScalar, FUNCTOR >					\
  OPERATOR(const FieldExpr<E>& op1, double op2) {			\
  return FieldBinOp<E, FieldScalar, FUNCTOR >				\
      (op1.get(), FieldScalar(op2), FUNCTOR());				\
  }									\
  									\
  template <typename E>							\
  FieldBinOp<FieldScalar, E, FUNCTOR >					\
  OPERATOR(double op1, const FieldExpr<E>& op2) {			\
    return FieldBinOp<FieldScalar, E, FUNCTOR >				\
      {FieldScalar(op1), op2.get(), FUNCTOR()};				\
  }									\

// Base class for scalar field expressions
template <typename E> struct FieldExpr {
  // call operator() on the base type E
  template <int N>
  double operator()(const SVector<N>& p) const {
    return static_cast<const E&>(*this)(p);
  }

  // get underyling type composing the expression node
  const E& get() const { return static_cast<const E&>(*this); }
};

// an expression node representing a scalar value (double, int, ... even single valued variables)
class FieldScalar : public FieldExpr<FieldScalar> {
private:
  double value_;
public:
  FieldScalar(double value) : value_(value) { }
  
  // call operator
  template <int N>
  double operator()(const SVector<N>& p) const { return value_; };
};

// expression template based arithmetic
template <typename OP1, typename OP2, typename BinaryOperation>
class FieldBinOp : public FieldExpr<FieldBinOp<OP1, OP2, BinaryOperation>> {
private:
  typename std::remove_reference<OP1>::type op1_;   // first  operand
  typename std::remove_reference<OP2>::type op2_;   // second operand
  BinaryOperation f_;                               // operation to apply

public:
  // constructor
  FieldBinOp(const OP1& op1, const OP2& op2, BinaryOperation f) : op1_(op1), op2_(op2), f_(f) { };

  // call operator, performs the expression evaluation
  template <int N>
  double operator()(const SVector<N>& p) const{
    return f_(op1_(p), op2_(p));
  }
};

DEF_FIELD_EXPR_OPERATOR(operator+, std::plus<>      )
DEF_FIELD_EXPR_OPERATOR(operator-, std::minus<>     )
DEF_FIELD_EXPR_OPERATOR(operator*, std::multiplies<>)

}}

#endif // __SCALAR_FIELD_EXPRESSIONS_H__