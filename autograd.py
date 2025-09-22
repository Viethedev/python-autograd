import math

class Gradable:
    def __init__(self, value=0):
        self.__value__ = value
        self.__grad__ = {self : 1}

    def __add__(self, other):
        if isinstance(other, Gradable):
            parent = Gradable(self.__value__ + other.__value__)
            for child, grad in other.__grad__.items():
                parent.__grad__[child] = parent.__grad__.get(child, 0) + grad
        else:
            parent = Gradable(self.__value__ + other)                 
        for child, grad in self.__grad__.items():
            parent.__grad__[child] = parent.__grad__.get(child, 0) + grad            
        return parent

    def __radd__(self, other):
        # Don’t call this dunder directly. It’s only for handling constant + Value 
        if not isinstance(other, Gradable):
            parent = Gradable(self.__value__ + other)     
            for child, grad in self.__grad__.items():
                parent.__grad__[child] = parent.__grad__.get(child, 0) + grad            
        return parent

    def __iadd__(self, other):
        if not isinstance(other, Gradable):
            self.__value__ += other
        return self

    def __sub__(self, other):
        if isinstance(other, Gradable):
            parent = Gradable(self.__value__ - other.__value__)
            for child, grad in other.__grad__.items():
                parent.__grad__[child] = parent.__grad__.get(child, 0) - grad
        else:
            parent = Gradable(self.__value__ - other)                
        for child, grad in self.__grad__.items():
            parent.__grad__[child] = parent.__grad__.get(child, 0) + grad            
        return parent

    def __rsub__(self, other):
        # Don’t call this dunder directly. It’s only for handling constant - Value 
        if not isinstance(other, Gradable):
            parent = Gradable(other - self.__value__)    
            for child, grad in self.__grad__.items():
                parent.__grad__[child] = parent.__grad__.get(child, 0) - grad            
        return parent

    def __isub__(self, other):
        if not isinstance(other, Gradable):
            self.__value__ -= other
        return self

    def __mul__(self, other):
        if isinstance(other, Gradable):
            parent = Gradable(self.__value__ * other.__value__)
            for child, grad in self.__grad__.items():
                parent.__grad__[child] = parent.__grad__.get(child, 0) + grad * other.__value__
            for child, grad in other.__grad__.items():
                parent.__grad__[child] = parent.__grad__.get(child, 0) + grad * self.__value__
        else:
            parent = Gradable(self.__value__ * other)
            for child, grad in self.__grad__.items():
                parent.__grad__[child] = parent.__grad__.get(child, 0) + grad * other
        return parent

    def __rmul__(self, other):
        # Don’t call this dunder directly. It’s only for handling constant * Value
        if not isinstance(other, Gradable):
            parent = Gradable(self.__value__ * other)
            for child, grad in self.__grad__.items():
                parent.__grad__[child] = parent.__grad__.get(child, 0) + grad * other
        return parent

    def __imul__(self, other):
        if not isinstance(other, Gradable):
            self.__value__ *= other
        return self
            
    def __truediv__(self, other):
        if isinstance(other, Gradable):
            parent = Gradable(self.__value__ / other.__value__)
            lhs_grad = 1 / other.__value__
            rhs_grad = -parent.__value__ / other.__value__
            for child, grad in self.__grad__.items():
                parent.__grad__[child] = parent.__grad__.get(child, 0) + grad * lhs_grad
            for child, grad in other.__grad__.items():
                parent.__grad__[child] = parent.__grad__.get(child, 0) + grad * rhs_grad
        else:
            parent = Gradable(self.__value__ / other)
            for child, grad in self.__grad__.items():
                parent.__grad__[child] = parent.__grad__.get(child, 0) + grad / other
        return parent

    def __rtruediv__(self, other):
        # Don’t call this dunder directly. It’s only for handling constant / Value 
        if not isinstance(other, Gradable):
            parent = Gradable(other / self.__value__)
            rhs_grad = -parent.__value__ / self.__value__
            for child, grad in self.__grad__.items():
                parent.__grad__[child] = parent.__grad__.get(child, 0) + grad * rhs_grad
        return parent

    def __itruediv__(self, other):
        if not isinstance(other, Gradable):
            self.__value__ /= other
        return self

    def __floordiv__(self, other):
        # The gradient is zero almost everywhere, undefined at integer discontinuities.
        if isinstance(other, Gradable):
            parent = Gradable(self.__value__ // other.__value__)
            for child in other.__grad__:
                parent.__grad__[child] = parent.__grad__.get(child, 0)
        else:
            parent = Gradable(self.__value__ // other)
        for child in self.__grad__:
            parent.__grad__[child] = parent.__grad__.get(child, 0)
        return parent

    def __rfloordiv__(self, other):
        # Don’t call this dunder directly. It’s only for handling constant // Value 
        if not isinstance(other, Gradable):
            parent = Gradable(other // self.__value__)
            for child in self.__grad__:
                parent.__grad__[child] = parent.__grad__.get(child, 0)
        return parent

    def __ifloordiv__(self, other):
        if not isinstance(other, Gradable):
            self.__value__ //= other
        return self

    def __mod__(self, other):
        if isinstance(other, Gradable):
            parent = Gradable(self.__value__ % other.__value__)
            rhs_grad = -(self.__value__ // other.__value__)
            for child, grad in other.__grad__.items():
                parent.__grad__[child] = parent.__grad__.get(child, 0) + grad * rhs_grad
        else:
            parent = Gradable(self.__value__ % other)
        for child, grad in self.__grad__.items():
            parent.__grad__[child] = parent.__grad__.get(child, 0) + grad
        return parent

    def __rmod__(self, other):
        # Don’t call this dunder directly. It’s only for handling constant / Value 
        if not isinstance(other, Gradable):
            parent = Gradable(other % self.__value__)
            rhs_grad = -(other // self.__value__)
            for child, grad in self.__grad__.items():
                parent.__grad__[child] = parent.__grad__.get(child, 0) + grad * rhs_grad
        return parent

    def __imod__(self, other):
        if not isinstance(other, Gradable):
            self.__value__ %= other
        return self

    def __pow__(self, other):
        if isinstance(other, Gradable):
            parent = Gradable(self.__value__ ** other.__value__)
            lhs_grad = other.__value__ * parent.__value__ / self.__value__
            rhs_grad = parent.__value__ * math.log(self.__value__)
            for child, grad in self.__grad__.items():
                parent.__grad__[child] = parent.__grad__.get(child, 0) + grad * lhs_grad
            for child, grad in other.__grad__.items():
                parent.__grad__[child] = parent.__grad__.get(child, 0) + grad * rhs_grad
        else:
            parent = Gradable(self.__value__ ** other)
            lhs_grad = other * parent.__value__ / self.__value__
            for child, grad in self.__grad__.items():
                parent.__grad__[child] = parent.__grad__.get(child, 0) + grad * lhs_grad
        return parent

    def __rpow__(self, other):
        # Don’t call this dunder directly. It’s only for handling constant ** Value 
        if not isinstance(other, Gradable):
            parent = Gradable(other ** self.__value__)
            rhs_grad = parent.__value__ * math.log(other)
            for child, grad in self.__grad__.items():
                parent.__grad__[child] = parent.__grad__.get(child, 0) + grad * rhs_grad
        return parent

    def __ipow__(self, other):
        if not isinstance(other, Gradable):
            self.__value__ **= other
        return self

    def __pos__(self):
        parent = Gradable(self.__value__)
        for child, grad in self.__grad__.items():
            parent.__grad__[child] = parent.__grad__.get(child, 0) + grad
        return parent

    def __neg__(self):
        parent = Gradable(-self.__value__)
        for child, grad in self.__grad__.items():
            parent.__grad__[child] = parent.__grad__.get(child, 0) - grad
        return parent

    def __abs__(self):
        parent = Gradable(abs(self.__value__))
        local_grad = 1 if self.__value__ >= 0 else -1
        for child, grad in self.__grad__.items():
            parent.__grad__[child] = parent.__grad__.get(child, 0) + grad * local_grad
        return parent

    def __exp__(self):
        parent = Gradable(math.exp(self.__value__))
        for child, grad in self.__grad__.items():
            parent.__grad__[child] = parent.__grad__.get(child, 0) + grad * parent.__value__
        return parent

    def __sin__(self):  # sine of angle in radian
        parent = Gradable(math.sin(self.__value__))
        local_grad = math.cos(self.__value__)
        for child, grad in self.__grad__.items():
            parent.__grad__[child] = parent.__grad__.get(child, 0) + grad * local_grad
        return parent

    def __cos__(self):  # cosine of angle in radian
        parent = Gradable(math.cos(self.__value__))
        local_grad = -math.sin(self.__value__)
        for child, grad in self.__grad__.items():
            parent.__grad__[child] = parent.__grad__.get(child, 0) + grad * local_grad
        return parent

    def __tan__(self):  # tangent of angle in radian (MathError if value = pi / 2 + k * 2 * pi)
        parent = Gradable(math.tan(self.__value__))
        local_grad = 1 / (math.cos(self.__value__) * math.cos(self.__value__)) 
        for child, grad in self.__grad__.items():
            parent.__grad__[child] = parent.__grad__.get(child, 0) + grad * local_grad
        return parent
    
    def __log__(self, base=math.e):
        if base == math.e:
            parent = Gradable(math.log(self.__value__))
            local_grad = 1 / self.__value__
        else:
            parent = Gradable(math.log(self.__value__, base))
            local_grad = 1 / (self.__value__ * math.log(base))

        for child, grad in self.__grad__.items():
            parent.__grad__[child] = parent.__grad__.get(child, 0) + grad * local_grad
        return parent

    def __str__(self): return str(self.__value__)

    def __repr__(self): return str(self.__value__)

    def __int__(self): return int(self.__value__)

    def __float__(self): return float(self.__value__)

    def __format__(self, spec): return format(self.__value__, spec)

    def __round__(self, digit:int=0):
        # The gradient is zero almost everywhere, undefined at integer discontinuities.
        parent = Gradable(round(self.__value__, digit))
        for child in self.__grad__:
            parent.__grad__[child] = parent.__grad__.get(child, 0)
        return parent

    def __trunc__(self):
        # The gradient is zero almost everywhere, undefined at integer discontinuities.
        parent = Gradable(math.trunc(self.__value__))
        for child in self.__grad__:
            parent.__grad__[child] = parent.__grad__.get(child, 0)
        return parent

    def __floor__(self):
        # The gradient is zero almost everywhere, undefined at integer discontinuities.
        parent = Gradable(math.floor(self.__value__))
        for child in self.__grad__:
            parent.__grad__[child] = parent.__grad__.get(child, 0)
        return parent

    def __ceil__(self):
        # The gradient is zero almost everywhere, undefined at integer discontinuities.
        parent = Gradable(math.ceil(self.__value__))
        for child in self.__grad__:
            parent.__grad__[child] = parent.__grad__.get(child, 0)
        return parent

    def grad(self, wrt, relation_alert=True): return self.__grad__.get(wrt, None if relation_alert else 0) # Get gradient locally

def sin(value): return value.__sin__() if isinstance(value, Gradable) else math.sin(value)

def cos(value): return value.__cos__() if isinstance(value, Gradable) else math.cos(value)

def tan(value): return value.__tan__() if isinstance(value, Gradable) else math.tan(value)

def exp(value): return value.__exp__() if isinstance(value, Gradable) else math.exp(value)

def log(value, base=math.e): return value.__log__(base) if isinstance(value, Gradable) else math.log(value, base)

# Special differentiate function
def dv(y, wrt, relation_alert=True): return y.__grad__.get(wrt, None if relation_alert else 0)

# A demo to show you how to interact with Values, get derivatives and configure relation_alert flag
if __name__ == "__main__":  
    x = Gradable(2.0)
    y = Gradable(1)
    f = 1 + x + 2 - x * 2 * x + sin(x) + exp(x) ** 4
    print(f"{x = }")                                # x = 2.0
    print(f"f(2) = {f:.6f}")                        # f(2) = 2978.867284
    print(f"df/dx at x=2 = {dv(f, x):.6f}")         # df/dx at x=2 = df/dx at x=2 = 11916.415801
    print(f"df/dy at y=1 = {dv(f, y)}")             # df/dy at y=1 = None
    print(f"df/dy at y=1 = {dv(f, y, False)}")      # df/dy at y=1 = 0
