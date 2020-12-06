import torch
import numpy

# ---------------
# calc gradiants:
print("calc gradiants:")
x = torch.ones(2, 2, requires_grad=True)
print(f"x: {x}")

y = x + 2
print(f"y: {y}")

z = y * y * 3
print(f"z: {z}")

out = z.mean()
print(f"out: {out}")

# Let’s backprop now. Because out contains a single scalar, out.backward() is equivalent to out.backward(torch.tensor(1.)).
out.backward()

# Now let's check the grads
print(x.grad)

# ------------------------
# vector-Jacobian product:
print("vector-Jacobian product:")
x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print(y)

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
print(x.grad)

# ---------------------------
# set requires grad to false:
print("set requires grad to false:")
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)

print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())
