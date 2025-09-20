<p align="center">
  <img src="python_autograd_logo.png" alt="Autograd Logo" width="200"/>
</p>

# python-autograd
A tiny automatic differentiation engine written in pure Python.  
Built for **learning**, not performance — and it has some unique twists that make the math feel natural.

## 🌟 Unique Features
### Use **variables as differentiation targets**: no more character-thingy!
  ```python
  dv(f, x)   # instead of dv(f, "x")
  ```
### Relation-aware derivatives: allow you to receive relation alert or not!
  ```python
  x = Value(1)
  y = Value(2)
  z = x + x
  print(f"dz/dy = {dv(z, y)}")
  ```
  Output
  ```
  dz/dy = None
  ```

  But if you set relation_alert=False:
  ```python
  x = Value(1)
  y = Value(2)
  z = x + x
  print(f"dz/dy = {dv(z, y, relation_alert=False)}")
  ```
  Output
  ```
  dz/dy = 0
  ```
## Quick Example
```python
from autograd import Value, sin, exp, dv

x = Value(2.0)
y = Value(1.0)

f = sin(x ** 2) * exp(x) # f(x) = sin(x²) * e^x

print("f(x) =", f.value)
print("df/dx =", dv(f, x))
print("df/dy =", dv(f, y))
```
Output
```
f(x) = -5.592056
df/dx = -24.911294
df/dy = None   # no relation between f and y
```
## Why this is different
Other autodiff libraries hide the mechanics.
Here, every derivative is explicitly propagated and stored.

### That means:
  You can literally “see” the chain rule at work.
  You’ll get a clear error if you ask for ∂f/∂z when z has no path to f.
  Perfect for classrooms, tutorials, and self-learners who want to understand.
