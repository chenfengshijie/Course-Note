import numpy as np

def f(x):
    return x[0] - x[1] + 2 * x[0] ** 2 + 2 * x[0] * x[1] + x[1] ** 2

def df(x):
    return np.array([1 + 4 * x[0] + 2 * x[1], -1 + 2 * x[0] + 2 * x[1]])


# Conjugate gradient method
def conjugate_gradient(A, b, x0, eps=1e-6):
    r = b - A @ x0
    d = r.copy()
    x = x0.copy()
    while np.linalg.norm(r) > eps:
        alpha = r.T @ r / (d.T @ A @ d)
        x += alpha * d
        r_next = r - alpha * A @ d
        beta = r_next.T @ r_next / (r.T @ r)
        d = r_next + beta * d
        r = r_next
    return x

# Hessian matrix
H = np.array([[4, 2], [2, 2]],dtype=np.float32)

x0 = np.array([0, 0],dtype=np.float32)
# Right-hand side of the linear equation Ax = b
b = -df(x0)

x = conjugate_gradient(H, b, x0)
print("Optimal value of x:", x)
print("Minimum value of f:", f(x))