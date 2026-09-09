"""
stage_03_stage_03_local_derivatives/code.py
In this stage I implement local derivatives for the
basic operations of addition and multiplication to fill in the gap
_backward left in stage 02. 
The local derivatives are implemented in the _backward method of the Value class.
"""

from __future__ import annotations

from typing import Union

from dlfs import stage_import


Stage2_Value = stage_import("stage_02", "Value")

Number = Union[int, float]

class Value(Stage2_Value):

    def __add__(self, other: Union[Value,Number]) -> Value:
        """
        coerce other to Value if it is a number, 
        then call the superclass __add__ method.
        define _backward() for addition. 
        self.grad+=other.grad, other.grad+=out.grad
        """
        other = other if isinstance(other, type(self)) else type(self)(other)
        out = super().__add__(other)
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    def __mul__(self, other: Union[Value,Number]) -> Value:
        """
        coerce other to Value if it is a number, 
        then call the superclass __mul__ method.
        define _backward() for multiplication. 
        self.grad+=other.data*out.grad, other.grad+=self.data*out.grad
        """
        other = other if isinstance(other, type(self)) else type(self)(other)
        out = super().__mul__(other)
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward =_backward
        return out

    def __pow__(self, other: Number) -> Value:
        """
        coerce other to Value if it is a number, 
        then call the superclass __pow__ method.
        define _backward() for exponentiation. 
        self.grad+=other.data*self.data**(other.data-1)*out.grad
        """
        if not isinstance(other, (int, float)):
            raise TypeError("Exponent must be a numeric type (int or float).")

        #We do not coerce other to Value here because stage 02 only supports numeric exponents.
        out = super().__pow__(other)
        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward
        return out

    def _set_backward(self, backward):
        """
        Set the _backward method for the Value object.
        This is used to define the local derivative for each operation.
        """
        self._backward = backward
        return self

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad= {self.grad})"