import numpy as np
import matplotlib.pyplot as plt

def ackley(x, a=20, b=0.2, c=2*np.pi):
    sum_sq = np.sum(x**2,axis=0)
    sum_cos = np.sum(np.cos(c * x),axis=0)
    # print('sum-sq',sum_sq)
    # print('sum-cos',sum_cos)
    d = len(x)
    return -a * np.exp(-b * np.sqrt(sum_sq / d)) - np.exp(sum_cos / d) + a + np.exp(1)

def d_ackley(x, a=20, b=0.2, c=2*np.pi):
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(c * x))
    d = len(x)
    factor = -a * b * np.exp(-b * np.sqrt(sum_sq / d)) / np.sqrt(sum_sq / d)
    return factor * x - np.exp(sum_cos / d) * np.sin(c * x)

def booth(x):
    return (x[0] + 2 * x[1] - 7)**2 + (2 * x[0] + x[1] - 5)**2

def d_booth(x):
    return np.array([10*x[0] + 8*x[1] - 34, 8*x[0] + 10*x[1] - 38])

def branin(x, a=1, b=5.1/(4*np.pi**2), c=5/np.pi, r=6, s=10, t=1/(8*np.pi)):
    term1 = x[1] - b * x[0]**2 + c * x[0] - r
    return a * term1**2 + s * (1 - t) * np.cos(x[0]) + s

def d_branin(x, a=1, b=5.1/(4*np.pi**2), c=5/np.pi, r=6, s=10, t=1/(8*np.pi)):
    term1 = x[1] - b * x[0]**2 + c * x[0] - r
    dx = 2*a*term1*(-2*b*x[0] + c) - s*(1-t)*np.sin(x[0])
    dy = 2*a*term1
    return np.array([dx, dy])

def rosenbrock_banana(x, a=1, b=5):
    return (a - x[0])**2 + b * (x[1] - x[0]**2)**2

def d_rosenbrock_banana(x, a=1, b=5):
    dx = -2*(a - x[0]) - 4*b*x[0]*(x[1] - x[0]**2)
    dy = 2*b*(x[1] - x[0]**2)
    return np.array([dx, dy])

def wheeler(x, a=1.5):
    return -np.exp(-(x[0]*x[1] - a)**2 - (x[1] - a)**2)

def d_wheeler(x, a=1.5):
    common_factor = 2 * np.exp(-(x[0]*x[1] - a)**2 - (x[1] - a)**2)
    dx = common_factor * (x[0] * x[1] - a) * x[1]
    dy = common_factor * ((x[0] * x[1] - a) * x[0] + (x[1] - a))
    return np.array([dx, dy])

def golden_section_search(f, a, b, delta):
    t = (np.sqrt(5) - 1) / 2
    x1, x2 = a + (1 - t) * (b - a), a + t * (b - a)
    f1, f2 = f(x1), f(x2)
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

def fr(f, df, x):
    ans = [x]
    d = -df(x)
    while True:
        lambda_ = golden_section_search(lambda lambda_: f(x + lambda_ * d), 0, 1, 1e-8)
        s = lambda_ * d
        if np.linalg.norm(s) < 1e-6:
            return x, ans
        x = x + s
        ans.append(x)
        beta = np.dot(df(x).T, df(x)) / np.dot(df(x - s).T, df(x - s))
        d = -df(x) + beta * d

def draw_function_optimization(f, df, x_range, y_range, x0, levels, title):
    X, Y = np.meshgrid(x_range, y_range)
    Z = f(np.array([X, Y]))
    # print('z',Z.ndim)
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contour(X, Y, Z, levels=levels, cmap='coolwarm')
    ax.clabel(contour, inline=True, fontsize=8)
    ax.set_title(title)
    min_point, ans = fr(f, df, x0)
    x_list, y_list = zip(*ans)
    ax.plot(x_list, y_list, '-o', color='black')
    ax.plot(min_point[0], min_point[1], 'x', markersize=10, color='red')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()

if __name__ == "__main__":
    # Ackley function optimization using FR
    x_range = np.linspace(-30, 30, 100)
    y_range = np.linspace(-30, 30, 100)
    levels = 10
    x0 = np.array([-10, 10])
    draw_function_optimization(ackley, d_ackley, x_range, y_range, x0, levels, 'Ackley Function using FR')

    # Booth function optimization using FR
    x_range = np.linspace(-10, 10, 100)
    y_range = np.linspace(-10, 10, 100)
    levels = 100
    x0 = np.array([-5, -5])
    draw_function_optimization(booth, d_booth, x_range, y_range, x0, levels, 'Booth Function using FR')

    # Branin function optimization using FR
    x_range = np.linspace(-5, 10, 100)
    y_range = np.linspace(0, 15, 100)
    levels = 100
    x0 = np.array([2.5, 7.5])
    draw_function_optimization(branin, d_branin, x_range, y_range, x0, levels, 'Branin Function using FR')

    # Rosenbrock function optimization using FR
    x_range = np.linspace(-2, 2, 100)
    y_range = np.linspace(-1, 3, 100)
    levels = 80
    x0 = np.array([0, -1])
    draw_function_optimization(rosenbrock_banana, d_rosenbrock_banana, x_range, y_range, x0, levels, 'Rosenbrock Banana Function using FR')

    # Wheeler function optimization using FR
    x_range = np.linspace(0, 5, 100)
    y_range = np.linspace(0, 5, 100)
    levels = 20
    x0 = np.array([1, 1])
    draw_function_optimization(wheeler, d_wheeler, x_range, y_range, x0, levels, 'Wheeler Function using FR')