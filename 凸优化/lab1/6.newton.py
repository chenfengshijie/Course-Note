import numpy as np

def f(x):
    return x[0]**2 + x[1]**2 - x[0]*x[1] - 10*x[0] - 4*x[1] + 60

def df(x):
    return np.array([2*x[0] - x[1] - 10, 2*x[1] - x[0] - 4])

def golden_section_search(f, a, b, delta):
    t = (np.sqrt(5) - 1) / 2
    x1 = a + (1 - t) * (b - a)
    x2 = a + t * (b - a)
    f1 = f(x1)
    f2 = f(x2)
    while b - a >= delta:
        if f1 > f2:
            a, x1, f1 = x1, x2, f2
            x2 = a + t * (b - a)
            f2 = f(x2)
        else:
            b, x2, f2 = x2, x1, f1
            x1 = a + (1 - t) * (b - a)
            f1 = f(x1)
    return (a + b) / 2

def quasi_newton_method(f, df, x, method='DFP', tol=1e-6):
    H = np.eye(len(x))
    while True:
        d = -np.dot(H, df(x))
        lamda = golden_section_search(lambda lamda: f(x + lamda * d), 0, 1, tol)
        s = lamda * d
        y = df(x + s) - df(x)
        if method == 'DFP':
            Hs = np.dot(H, s)
            H += np.outer(s, s) / np.dot(s, y) - np.outer(Hs, Hs) / np.dot(s, Hs)
        elif method == 'BFGS':
            rho = 1.0 / np.dot(y, s)
            I = np.eye(len(x))
            V = I - rho * np.outer(s, y)
            H = np.dot(np.dot(V, H), V.T) + rho * np.outer(s, s)
        if np.linalg.norm(s) < tol:
            return x, f(x)
        x += s

def fr_method(f, df, x, tol=1e-6):
    d = -df(x)
    while True:
        lamda = golden_section_search(lambda lamda: f(x + lamda * d), 0, 1, tol)
        s = lamda * d
        if np.linalg.norm(s) < tol:
            return x, f(x)
        x += s
        df_x = df(x)
        beta = np.dot(df_x, df_x) / np.dot(df(x - s), df(x - s))
        d = -df_x + beta * d

if __name__ == "__main__":
    x0 = np.array([0, 0],dtype=np.float64)
    print("(a) DFP算法：", quasi_newton_method(f, df, x0, method='DFP'))
    print("(b) BFGS算法：", quasi_newton_method(f, df, x0, method='BFGS'))
    print("(c) FR算法：", fr_method(f, df, x0))