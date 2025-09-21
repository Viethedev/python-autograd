import math

class Variable:
    def __init__(self, value=0):
        self.__value__ = value
        self.__grad__ = {}

    def __add__(self, other):
        if isinstance(other, Variable):
            parent = Variable(self.__value__ + other.__value__)
            parent.__grad__[other] = 1
            for child, grad in other.__grad__.items():
                parent.__grad__[child] = parent.__grad__.get(child, 0) + grad
        else:
            parent = Variable(self.__value__ + other)            
        parent.__grad__[self] = 1     
        for child, grad in self.__grad__.items():
            parent.__grad__[child] = parent.__grad__.get(child, 0) + grad            
        return parent

    def __radd__(self, other):
        # Don't use the dunder yourself, for handling constant + Value only
        if not isinstance(other, Variable):
            parent = Variable(self.__value__ + other)
            parent.__grad__[self] = 1     
            for child, grad in self.__grad__.items():
                parent.__grad__[child] = parent.__grad__.get(child, 0) + grad            
        return parent

    def __sub__(self, other):
        if isinstance(other, Variable):
            parent = Variable(self.__value__ - other.__value__)
            parent.__grad__[other] = -1
            for child, grad in other.__grad__.items():
                parent.__grad__[child] = parent.__grad__.get(child, 0) - grad
        else:
            parent = Variable(self.__value__ - other)            
        parent.__grad__[self] = 1     
        for child, grad in self.__grad__.items():
            parent.__grad__[child] = parent.__grad__.get(child, 0) + grad            
        return parent

    def __rsub__(self, other):
        # Don't use the dunder yourself, for handling constant + Value only
        if not isinstance(other, Variable):
            parent = Variable(-self.__value__ + other)
            parent.__grad__[self] = -1     
            for child, grad in self.__grad__.items():
                parent.__grad__[child] = parent.__grad__.get(child, 0) - grad            
        return parent

    def __mul__(self, other):
        if isinstance(other, Variable):
            parent = Variable(self.__value__ * other.__value__)
            parent.__grad__[self] = other.__value__
            parent.__grad__[other] = self.__value__
            for child, grad in self.__grad__.items():
                parent.__grad__[child] = parent.__grad__.get(child, 0) + grad * other.__value__
            for child, grad in other.__grad__.items():
                parent.__grad__[child] = parent.__grad__.get(child, 0) + grad * self.__value__
        else:
            parent = Variable(self.__value__ * other)
            parent.__grad__[self] = other
            for child, grad in self.__grad__.items():
                parent.__grad__[child] = parent.__grad__.get(child, 0) + grad * other
        return parent

    def __rmul__(self, other):
        # Don't use the dunder yourself, for handling constant * Value only
        if not isinstance(other, Variable):
            parent = Variable(self.__value__ * other)
            parent.__grad__[self] = other
            for child, grad in self.__grad__.items():
                parent.__grad__[child] = parent.__grad__.get(child, 0) + grad * other
        return parent

    def __truediv__(self, other):
        if isinstance(other, Variable):
            parent = Variable(self.__value__ / other.__value__)
            parent.__grad__[self] = 1 / other.__value__
            parent.__grad__[other] = -parent.__value__ / other.__value__
            for child, grad in self.__grad__.items():
                parent.__grad__[child] = parent.__grad__.get(child, 0) + grad / other.__value__
            for child, grad in other.__grad__.items():
                parent.__grad__[child] = parent.__grad__.get(child, 0) - grad * parent.__value__ / other.__value__
        else:
            parent = Variable(self.__value__ / other)
            parent.__grad__[self] = 1 / other
            for child, grad in self.__grad__.items():
                parent.__grad__[child] = parent.__grad__.get(child, 0) + grad / other
        return parent

    def __rtruediv__(self, other):
        # Don't use the dunder yourself, for handling constant / Value only
        if not isinstance(other, Variable):
            parent = Variable(other / self.__value__)
            parent.__grad__[self] = -parent.__value__ / other
            for child, grad in self.__grad__.items():
                parent.__grad__[child] = parent.__grad__.get(child, 0) - grad * parent.__value__ / other
        return parent

    def __floordiv__(self, other):
        # The gradient is zero almost everywhere, except at integer points where the function is discontinuous.
        if isinstance(other, Variable):
            parent = Variable(self.__value__ // other.__value__)
            parent.__grad__[other] = 0
            for child, grad in other.__grad__.items():
                parent.__grad__[child] = parent.__grad__.get(child, 0)
        else:
            parent = Variable(self.__value__ // other)
        parent.__grad__[self] = 0
        for child, grad in self.__grad__.items():
            parent.__grad__[child] = parent.__grad__.get(child, 0)
        return parent

    def __rfloordiv__(self, other):
        # Don't use the dunder yourself, for handling constant // Value only
        if not isinstance(other, Variable):
            parent = Variable(other // self.__value__)
            parent.__grad__[self] = 0
            for child, grad in self.__grad__.items():
                parent.__grad__[child] = parent.__grad__.get(child, 0)
        return parent

    def __mod__(self, other):
        if isinstance(other, Variable):
            parent = Variable(self.__value__ % other.__value__)
            parent.__grad__[other] = -(self.__value__ // other.__value__)
            for child, grad in other.__grad__.items():
                parent.__grad__[child] = parent.__grad__.get(child, 0) - grad * self.__value__ // other.__value__
        else:
            parent = Variable(self.__value__ / other)
        parent.__grad__[self] = 1
        for child, grad in self.__grad__.items():
            parent.__grad__[child] = parent.__grad__.get(child, 0) + grad
        return parent

    def __rmod__(self, other):
        # Don't use the dunder yourself, for handling constant / Value only
        if not isinstance(other, Variable):
            parent = Variable(other % self.__value__)
            parent.__grad__[self] = -(other // self.__value__)
            for child, grad in self.__grad__.items():
                parent.__grad__[child] = parent.__grad__.get(child, 0) - grad * other // self.__value__
        return parent

    def __pow__(self, other):
        if isinstance(other, Variable):
            parent = Variable(self.__value__ ** other.__value__)
            parent.__grad__[self] = other.__value__ * parent.__value__ / self.__value__
            parent.__grad__[other] = parent.__value__ * math.log(self.__value__)
            for child, grad in self.__grad__.items():
                parent.__grad__[child] = parent.__grad__.get(child, 0) + grad * other.__value__ * parent.__value__ / self.__value__
            for child, grad in other.__grad__.items():
                parent.__grad__[child] = parent.__grad__.get(child, 0) + grad * parent.__value__ + math.log(self.__value__)
        else:
            parent = Variable(self.__value__ ** other)
            parent.__grad__[self] = other * parent.__value__ / self.__value__
            for child, grad in self.__grad__.items():
                parent.__grad__[child] = parent.__grad__.get(child, 0) + grad * other * parent.__value__ / self.__value__
        return parent

    def __rpow__(self, other):
        # Don't use the dunder yourself, for handling constant ** Value only
        if not isinstance(other, Variable):
            parent = Variable(other ** self.value)
            parent.__grad__[self] = parent.__value__ * math.log(self.__value__)
            for child, grad in self.__grad__.items():
                parent.__grad__[child] = parent.__grad__.get(child, 0) + grad * parent.__value__ + math.log(self.__value__)
        return parent

    def __pos__(self):
        parent = self
        parent.__grad__[self] = 1
        for child, grad in self.__grad__.items():
            parent.__grad__[child] = parent.__grad__.get(child, 0) + grad
        return parent

    def __neg__(self):
        parent = Variable(-self.__value__)
        parent.__grad__[self] = -1
        for child, grad in self.__grad__.items():
            parent.__grad__[child] = parent.__grad__.get(child, 0) - grad
        return parent

    def __abs__(self):
        parent = Variable(abs(self.__value__))
        parent.__grad__[self] = 1 if self.__value__ >= 0 else -1
        for child, grad in self.__grad__.items():
            parent.__grad__[child] = parent.__grad__.get(child, 0) + (grad if self.__value__ >= 0 else -grad)
        return parent

    def __exp__(self):
        parent = Variable(math.exp(self.__value__))
        parent.__grad__[self] = parent.__value__
        for child, grad in self.__grad__.items():
            parent.__grad__[child] = parent.__grad__.get(child, 0) + grad * parent.__value__
        return parent

    def __sin__(self):  # sine of angle in radian
        parent = Variable(math.sin(self.__value__))
        parent.__grad__[self] = math.cos(self.__value__)
        for child, grad in self.__grad__.items():
            parent.__grad__[child] = parent.__grad__.get(child, 0) + grad * math.cos(self.__value__)
        return parent

    def __cos__(self):  # cosine of angle in radian
        parent = Variable(math.cos(self.__value__))
        parent.__grad__[self] = -math.sin(self.__value__)
        for child, grad in self.__grad__.items():
            parent.__grad__[child] = parent.__grad__.get(child, 0) - grad * math.sin(self.__value__)
        return parent

    def __tan__(self):  # tangent of angle in radian (MathError if value = pi / 2 + k * 2 * pi)
        parent = Variable(math.tan(self.__value__))
        parent.__grad__[self] = 1 / (math.cos(self.__value__) * math.cos(self.__value__)) 
        for child, grad in self.__grad__.items():
            parent.__grad__[child] = parent.__grad__.get(child, 0) - grad / (math.cos(self.__value__) * math.cos(self.__value__)) 
        return parent

    def __str__(self): return self.__value__

    def __int__(self): return int(self.__value__)

    def __float__(self): return float(self.__value__)

    def __format__(self, spec): return format(self.__value__, spec)

    def makeint(self): self.__value__ = int(self.__value__) # Make the value integer-rounded

    def grad(self, wrt, relation_alert=True): return self.__grad__.get(wrt, None if relation_alert else 0) # Get gradient locally

def sin(value): return value.__sin__() if isinstance(value, Variable) else math.sin(value)

def cos(value): return value.__cos__() if isinstance(value, Variable) else math.cos(value)

def tan(value): return value.__tan__() if isinstance(value, Variable) else math.tan(value)

def exp(value): return value.__exp__() if isinstance(value, Variable) else math.exp(value)

# Special differentiate function
def dv(y, wrt, relation_alert=True): return y.__grad__.get(wrt, None if relation_alert else 0)

# A demo to show you how to interact with Values, get derivatives and configure relation_alert flag 
if __name__ == "__main__":  
    x = Variable(2.0)
    y = Variable(1)
    f = 1 + x + 2 - x * 2 * x + sin(x) + exp(x) ** 4

    print(f"f(2) = {f:.6f}")                  # f(2) = 2978.867284
    print(f"df/dx at x=2 = {dv(f, x):.6f}")         # df/dx at x=2 = df/dx at x=2 = 11916.415801
    print(f"df/dy at y=1 = {dv(f, y)}")             # df/dy at y=1 = None
    print(f"df/dy at y=1 = {dv(f, y, False)}")      # df/dy at y=1 = 0

