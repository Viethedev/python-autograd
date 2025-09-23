import math

class Gradable:
    def __init__(self, *anonymous, **identified):
        if anonymous and identified:
            raise TypeError("Gradable() got multiple values for argument")
        
        if len(anonymous) == 1 and not identified:
            self.name = None
            self._value = float(anonymous[0])
        
        elif len(identified) == 1 and not anonymous:
            self.name, value = list(identified.items())[0]
            self._value = float(value)
        
        elif len(anonymous) == 0 and len(identified) == 0:
            raise TypeError("Gradable() missing 1 required argument")
        
        else:
            raise TypeError("Gradable() takes exactly one argument (positional or keyword)")
        self._grad = {self: 1}

    def __add__(self, other):
        if isinstance(other, Gradable):
            parent = Gradable(self._value + other._value)
            for child, grad in other._grad.items():
                parent._grad[child] = parent._grad.get(child, 0) + grad
        else:
            parent = Gradable(self._value + other)                 
        for child, grad in self._grad.items():
            parent._grad[child] = parent._grad.get(child, 0) + grad            
        return parent

    def __radd__(self, other):
        # Don’t call this dunder directly. It’s only for handling constant + Value 
        if not isinstance(other, Gradable):
            parent = Gradable(self._value + other)     
            for child, grad in self._grad.items():
                parent._grad[child] = parent._grad.get(child, 0) + grad            
        return parent

    def __iadd__(self, other):
        # Inplace with int or float only
        if not isinstance(other, Gradable):
            self._value += other
        return self

    def __sub__(self, other):
        if isinstance(other, Gradable):
            parent = Gradable(self._value - other._value)
            for child, grad in other._grad.items():
                parent._grad[child] = parent._grad.get(child, 0) - grad
        else:
            parent = Gradable(self._value - other)                
        for child, grad in self._grad.items():
            parent._grad[child] = parent._grad.get(child, 0) + grad            
        return parent

    def __rsub__(self, other):
        # Don’t call this dunder directly. It’s only for handling constant - Value 
        if not isinstance(other, Gradable):
            parent = Gradable(other - self._value)    
            for child, grad in self._grad.items():
                parent._grad[child] = parent._grad.get(child, 0) - grad            
        return parent

    def __isub__(self, other):
        # Inplace with int or float only
        if not isinstance(other, Gradable):
            self._value -= other
        return self

    def __mul__(self, other):
        if isinstance(other, Gradable):
            parent = Gradable(self._value * other._value)
            for child, grad in self._grad.items():
                parent._grad[child] = parent._grad.get(child, 0) + grad * other._value
            for child, grad in other._grad.items():
                parent._grad[child] = parent._grad.get(child, 0) + grad * self._value
        else:
            parent = Gradable(self._value * other)
            for child, grad in self._grad.items():
                parent._grad[child] = parent._grad.get(child, 0) + grad * other
        return parent

    def __rmul__(self, other):
        # Don’t call this dunder directly. It’s only for handling constant * Value
        if not isinstance(other, Gradable):
            parent = Gradable(self._value * other)
            for child, grad in self._grad.items():
                parent._grad[child] = parent._grad.get(child, 0) + grad * other
        return parent

    def __imul__(self, other):
        # Inplace with int or float only
        if not isinstance(other, Gradable):
            self._value *= other
        return self
            
    def __truediv__(self, other):
        if isinstance(other, Gradable):
            parent = Gradable(self._value / other._value)
            lhs_grad = 1 / other._value
            rhs_grad = -parent._value / other._value
            for child, grad in self._grad.items():
                parent._grad[child] = parent._grad.get(child, 0) + grad * lhs_grad
            for child, grad in other._grad.items():
                parent._grad[child] = parent._grad.get(child, 0) + grad * rhs_grad
        else:
            parent = Gradable(self._value / other)
            for child, grad in self._grad.items():
                parent._grad[child] = parent._grad.get(child, 0) + grad / other
        return parent

    def __rtruediv__(self, other):
        # Don’t call this dunder directly. It’s only for handling constant / Value 
        if not isinstance(other, Gradable):
            parent = Gradable(other / self._value)
            rhs_grad = -parent._value / self._value
            for child, grad in self._grad.items():
                parent._grad[child] = parent._grad.get(child, 0) + grad * rhs_grad
        return parent

    def __itruediv__(self, other):
        # Inplace with int or float only
        if not isinstance(other, Gradable):
            self._value /= other
        return self

    def __floordiv__(self, other):
        # The gradient is zero almost everywhere, undefined at integer discontinuities.
        if isinstance(other, Gradable):
            parent = Gradable(self._value // other._value)
            for child in other._grad:
                parent._grad[child] = parent._grad.get(child, 0)
        else:
            parent = Gradable(self._value // other)
        for child in self._grad:
            parent._grad[child] = parent._grad.get(child, 0)
        return parent

    def __rfloordiv__(self, other):
        # Don’t call this dunder directly. It’s only for handling constant // Value 
        if not isinstance(other, Gradable):
            parent = Gradable(other // self._value)
            for child in self._grad:
                parent._grad[child] = parent._grad.get(child, 0)
        return parent

    def __ifloordiv__(self, other):
        # Inplace with int or float only
        if not isinstance(other, Gradable):
            self._value //= other
        return self

    def __mod__(self, other):
        if isinstance(other, Gradable):
            parent = Gradable(self._value % other._value)
            rhs_grad = -(self._value // other._value)
            for child, grad in other._grad.items():
                parent._grad[child] = parent._grad.get(child, 0) + grad * rhs_grad
        else:
            parent = Gradable(self._value % other)
        for child, grad in self._grad.items():
            parent._grad[child] = parent._grad.get(child, 0) + grad
        return parent

    def __rmod__(self, other):
        # Don’t call this dunder directly. It’s only for handling constant / Value 
        if not isinstance(other, Gradable):
            parent = Gradable(other % self._value)
            rhs_grad = -(other // self._value)
            for child, grad in self._grad.items():
                parent._grad[child] = parent._grad.get(child, 0) + grad * rhs_grad
        return parent

    def __imod__(self, other):
        # Inplace with int or float only
        if not isinstance(other, Gradable):
            self._value %= other
        return self

    def __pow__(self, other):
        if isinstance(other, Gradable):
            parent = Gradable(self._value ** other._value)
            lhs_grad = other._value * parent._value / self._value
            rhs_grad = parent._value * math.log(self._value)
            for child, grad in self._grad.items():
                parent._grad[child] = parent._grad.get(child, 0) + grad * lhs_grad
            for child, grad in other._grad.items():
                parent._grad[child] = parent._grad.get(child, 0) + grad * rhs_grad
        else:
            parent = Gradable(self._value ** other)
            lhs_grad = other * parent._value / self._value
            for child, grad in self._grad.items():
                parent._grad[child] = parent._grad.get(child, 0) + grad * lhs_grad
        return parent

    def __rpow__(self, other):
        # Don’t call this dunder directly. It’s only for handling constant ** Value 
        if not isinstance(other, Gradable):
            parent = Gradable(other ** self._value)
            rhs_grad = parent._value * math.log(other)
            for child, grad in self._grad.items():
                parent._grad[child] = parent._grad.get(child, 0) + grad * rhs_grad
        return parent

    def __ipow__(self, other):
        # Inplace with int or float only
        if not isinstance(other, Gradable):
            self._value **= other
        return self

    def __pos__(self):
        parent = Gradable(self._value)
        for child, grad in self._grad.items():
            parent._grad[child] = parent._grad.get(child, 0) + grad
        return parent

    def __neg__(self):
        parent = Gradable(-self._value)
        for child, grad in self._grad.items():
            parent._grad[child] = parent._grad.get(child, 0) - grad
        return parent

    def __abs__(self):
        parent = Gradable(abs(self._value))
        local_grad = 1 if self._value >= 0 else -1
        for child, grad in self._grad.items():
            parent._grad[child] = parent._grad.get(child, 0) + grad * local_grad
        return parent

    def __exp__(self):
        parent = Gradable(math.exp(self._value))
        for child, grad in self._grad.items():
            parent._grad[child] = parent._grad.get(child, 0) + grad * parent._value
        return parent

    def __sin__(self):  # sine of angle in radian
        parent = Gradable(math.sin(self._value))
        local_grad = math.cos(self._value)
        for child, grad in self._grad.items():
            parent._grad[child] = parent._grad.get(child, 0) + grad * local_grad
        return parent

    def __cos__(self):  # cosine of angle in radian
        parent = Gradable(math.cos(self._value))
        local_grad = -math.sin(self._value)
        for child, grad in self._grad.items():
            parent._grad[child] = parent._grad.get(child, 0) + grad * local_grad
        return parent

    def __tan__(self):  # tangent of angle in radian (MathError if value = pi / 2 + k * 2 * pi)
        parent = Gradable(math.tan(self._value))
        local_grad = 1 / (math.cos(self._value) * math.cos(self._value)) 
        for child, grad in self._grad.items():
            parent._grad[child] = parent._grad.get(child, 0) + grad * local_grad
        return parent
    
    def __log__(self, base=math.e):
        # Use int or float base only
        if base == math.e:
            parent = Gradable(math.log(self._value))
            local_grad = 1 / self._value
        else:
            parent = Gradable(math.log(self._value, base))
            local_grad = 1 / (self._value * math.log(base))

        for child, grad in self._grad.items():
            parent._grad[child] = parent._grad.get(child, 0) + grad * local_grad
        return parent

    def __str__(self): return str(self._value)

    def __repr__(self): return str(self._value)

    def __int__(self): return int(self._value)

    def __float__(self): return float(self._value)

    def __format__(self, spec): return format(self._value, spec)

    def __round__(self, digit:int=0):
        # The gradient is zero almost everywhere, undefined at integer discontinuities.
        parent = Gradable(round(self._value, digit))
        for child in self._grad:
            parent._grad[child] = parent._grad.get(child, 0)
        return parent

    def __trunc__(self):
        # The gradient is zero almost everywhere, undefined at integer discontinuities.
        parent = Gradable(math.trunc(self._value))
        for child in self._grad:
            parent._grad[child] = parent._grad.get(child, 0)
        return parent

    def __floor__(self):
        # The gradient is zero almost everywhere, undefined at integer discontinuities.
        parent = Gradable(math.floor(self._value))
        for child in self._grad:
            parent._grad[child] = parent._grad.get(child, 0)
        return parent

    def __ceil__(self):
        # The gradient is zero almost everywhere, undefined at integer discontinuities.
        parent = Gradable(math.ceil(self._value))
        for child in self._grad:
            parent._grad[child] = parent._grad.get(child, 0)
        return parent

    def grad(self, wrt, relation_alert=True): return self._grad.get(wrt, None if relation_alert else 0) # Get gradient locally

def sin(value): return value.__sin__() if isinstance(value, Gradable) else math.sin(value)

def cos(value): return value.__cos__() if isinstance(value, Gradable) else math.cos(value)

def tan(value): return value.__tan__() if isinstance(value, Gradable) else math.tan(value)

def exp(value): return value.__exp__() if isinstance(value, Gradable) else math.exp(value)

def log(value, base=math.e): return value.__log__(base) if isinstance(value, Gradable) else math.log(value, base)

# Special differentiate function & utility
def dv(gradable, wrt, relation_alert=True): return gradable._grad.get(wrt, None if relation_alert else 0)

def inspect_dv(gradable, wrt):
    answer = dv(gradable, wrt)
    second_part = "Dependant" if gradable.name == None else f"{gradable.name}"
    fourth_part = "Independant" if wrt.name == None else f"{wrt.name}"
    if answer == None:
        first_part = "No relation between "
        third_part = " and "
        last_part = ""
    else:
        first_part = "d"
        third_part = "/d"
        last_part = f"({fourth_part}={wrt}) = {answer}"
    print(first_part + second_part + third_part + fourth_part + last_part)
    return answer

# A demo to show you how to interact with Values, get derivatives and use dv() and inspect_dv()
if __name__ == "__main__":  
    x = Gradable(x=2.0)
    y = Gradable(y=1)
    f = 1 + x + 2 - x * 2 * x + sin(x) + exp(x) ** 4
    f.name = "f"
    print(f"{x = }")                                # x = 2.0
    print(f"f(2) = {f:.6f}")                        # f(2) = 2978.867284
    inspect_dv(f, x)                                # df/dx(x=2.0) = 11916.415801330368
    print(dv(f, y))                                 # None
    inspect_dv(f, Gradable(3.000))                  # No relation between f and Independant

