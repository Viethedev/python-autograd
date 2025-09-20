import math

class Value:
    def __init__(self, value=0):
        self.value = value
        self.grads = {}

    def __add__(self, other):
        parent = Value(self.value + other.value)
        parent.grads[self] = 1
        parent.grads[other] = 1
        for child, grad in self.grads.items():
            parent.grads[child] = parent.grads.get(child, 0) + grad
        for child, grad in other.grads.items():
            parent.grads[child] = parent.grads.get(child, 0) + grad
        return parent

    def __mul__(self, other):
        parent = Value(self.value * other.value)
        parent.grads[self] = other.value
        parent.grads[other] = self.value
        for child, grad in self.grads.items():
            parent.grads[child] = parent.grads.get(child, 0) + grad * other.value
        for child, grad in other.grads.items():
            parent.grads[child] = parent.grads.get(child, 0) + grad * self.value
        return parent

    def __pow__(self, power):
        parent = Value(self.value ** power)
        parent.grads[self] = power * (self.value ** (power - 1))
        for child, grad in self.grads.items():
            parent.grads[child] = parent.grads.get(child, 0) + grad * power * (self.value ** (power - 1))
        return parent

    def exp(self):
        parent = Value(math.exp(self.value))
        parent.grads[self] = parent.value
        for child, grad in self.grads.items():
            parent.grads[child] = parent.grads.get(child, 0) + grad * parent.value
        return parent

    def sin(self):
        parent = Value(math.sin(self.value))
        parent.grads[self] = math.cos(self.value)
        for child, grad in self.grads.items():
            parent.grads[child] = parent.grads.get(child, 0) + grad * math.cos(self.value)
        return parent

    def __str__(self):
        return self.value

def sin(value): return value.sin() if isinstance(value, Value) else math.sin(value)

def exp(value): return value.exp() if isinstance(value, Value) else math.exp(value)

def dv(y, wrt, relation_alert=True): return y.grads.get(wrt, None if relation_alert else 0)

if __name__ == "__main__":
    y = Value(1)
    x = Value(2.0)               # x = 2
    f = sin(x ** 2) * exp(x) # f(x) = sin(x^2) * e^x

    print(f"f(x) = {f.value:.6f}")
    print(f"df/dx at x=2 = {dv(f, x):.6f}")
    print(f"df/dy at y=1 = {dv(f, y)}")
