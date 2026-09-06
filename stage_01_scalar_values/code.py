class Value:
    """A scalar (stored as float) with arithmetic via operator overloading.

    Fields:
    - ``data``: the wrapped number (float).
    - ``grad``: 0.0 here; filled by the stage_04 ``.backward()`` pass.
    - ``_prev``: the set of operand ``Value``s this result was built from
      (``set()`` for a leaf). A set so a reused operand (``a * a``) is one parent.
    - ``_op``: a string label for the op that built this node (``''`` for a leaf).
    - ``_backward``: a no-op closure here; stage_03 installs the per-op gradient
      rule on it. Reserving the field now keeps the field set stable across stages.

    Operand ORDER is not stored: ``_prev`` is a set, which is fine because the
    stage_03 gradient closures capture each operand directly, so ``a - b`` and
    ``a / b`` know which side is which without the node remembering order.
    """
    def __init__(self, data, _children=(), _op=''):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)  
        self._op = _op
        self._backward = lambda: None  #will be set in stage_03

    
    def _make(self, data, _children, _op):
        """
        Makes a new Value node with the same data, but new children and op. 
        Used in operator overloads to ensure uniformity of the Value type.
        """
        return type(self)(data, _children, _op)

    def __repr__(self):
        """
        Returns a string representation of the Value object, showing its data.
        """
        return f"Value(data={self.data})"

    def __add__(self, other):
        """
        Overloads the + operator to add two Value objects. 
        Creates a new Value object with the sum of the data from both operands,
        and sets the current Value object and the other operand as its children.
        """
        other = other if isinstance(other, Value) else Value(other)
        return self._make(self.data + other.data, (self, other), '+')

    def __mul__(self, other):
        """
        Overloads the * operator to multiply two Value objects. 
        Creates a new Value object with the product of the data from both operands,
        and sets the current Value object and the other operand as its children.
        """
        other = other if isinstance(other, Value) else Value(other)
        return self._make(self.data * other.data, (self, other), '*')

    def __pow__(self, other):
        """
        Overloads the ** operator to raise a Value object to the power of another. 
        Creates a new Value object with the result of raising the data from the current
        Value object to the power of the data from the other operand, and sets both
        as its children.
        """
        if(not isinstance(other, (int, float))):
            raise TypeError("Exponent must be a numeric type (int or float).")
        other = other if isinstance(other, Value) else Value(other)
        return self._make(self.data ** other.data, (self, other), '**')
        

    def __sub__(self, other):
        """
        Overloads the - operator to subtract one Value object from another. 
        Creates a new Value object with the difference of the data from both operands,
        and sets the current Value object and the other operand as its children.
        """
        other = other if isinstance(other, Value) else Value(other)
        return self._make(self.data - other.data, (self, other), '-')

    def __truediv__(self, other):
        """
        Overloads the / operator to divide one Value object by another.
        Creates a new Value object with the result of dividing the data from the current
        Value object by the data from the other operand, and sets both as its children.
        """
        other = other if isinstance(other, Value) else Value(other)
        return self._make(self.data / other.data, (self, other), '/')

    def __neg__(self):
        """
        Overloads the unary - operator to negate a Value object. 
        Creates a new Value object with the negated data from the current Value object,
        and sets the current Value object as its child.
        """
        return self._make(-self.data, (self,), 'neg')
    
    # reflected operators: enable `2 * a`, `1 + a`, `3 - a`, `6 / a`

    def __radd__(self, other):
        """Return other + self."""
        return self + other  # addition is commutative
    
    def __rmul__(self, other):
        """Return other * self."""
        return self * other  # multiplication is commutative

    def __rsub__(self, other):
        """Return other - self (as other + (-self))."""
        return Value(other) - self
    def __rtruediv__(self, other):
        """Return other / self (as other * self ** -1)."""
        return Value(other) / self