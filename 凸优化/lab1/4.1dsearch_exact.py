import numpy as np


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


def fibonacci_search(f, a, b, delta):

    def fibonacci(n, computed={0: 0, 1: 1}):
        if n not in computed:
            computed[n] = fibonacci(n - 1, computed) + fibonacci(n - 2, computed)
        return computed[n]

    n = 1
    while fibonacci(n + 1) < (b - a) / delta:
        n += 1
    
    x1 = a + (fibonacci(n - 1) / fibonacci(n + 1)) * (b - a)
    x2 = a + (fibonacci(n) / fibonacci(n + 1)) * (b - a)
    f1 = f(x1)
    f2 = f(x2)
    
    for i in range(n - 1, 0, -1):
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = a + (fibonacci(i - 1) / fibonacci(i + 1)) * (b - a)
            f1 = f(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + (fibonacci(i) / fibonacci(i + 1)) * (b - a)
            f2 = f(x2)
    return (a + b) / 2


def binary_search(df, a, b, delta):

    while b - a >= delta:
        lmda = (a + b) / 2
        df_val = df(lmda)
        if df_val > 0:
            b = lmda
        elif df_val < 0:
            a = lmda
        else:
            return lmda
    return (a + b) / 2


def dichotomous_search(f, a, b, eps, delta):

    while b - a >= delta:
        mid = (a + b) / 2
        lmd = mid - eps
        miu = mid + eps
        if f(lmd) < f(miu):
            b = miu
        else:
            a = lmd
    return (a + b) / 2

def shubert_piyavskii(f, a, b, l, eps=1e-6, max_iter=100):
    # Initialize the list of potential minima points
    potential_minima = [(a, f(a)), (b, f(b))]
    
    # Function to calculate the lower bound of the function at point x
    def lower_bound(x, x1, y1, x2, y2):
        return max(min(y1 + l * (x - x1), y2 + l * (x - x2)), min(y1, y2))
    
    for _ in range(max_iter):
        potential_minima.sort(key=lambda x: x[0])
        
        max_diff = 0
        max_index = 0
        for i in range(len(potential_minima) - 1):
            x1, y1 = potential_minima[i]
            x2, y2 = potential_minima[i + 1]
            m = (y1 - y2) / (x1 - x2)
            x_new = (x1 + x2 - (y1 - y2) / l) / 2
            lb = lower_bound(x_new, x1, y1, x2, y2)
            diff = min(y1, y2) - lb
            if diff > max_diff:
                max_diff = diff
                max_index = i
                x_candidate = x_new

        if max_diff < eps:
            break
        
        y_candidate = f(x_candidate)
        potential_minima.append((x_candidate, y_candidate))

    potential_minima.sort(key=lambda x: x[1])
    return potential_minima[0]

def f_a(x):
    return 2 * x ** 2 - x - 1


def df_a(x):
    return 4 * x - 1


def f_b(x):
    return 3 * x ** 2 - 21.6 * x - 1


def df_b(x):
    return 6 * x - 21.6


if __name__ == "__main__":
    print("(x)=2x^2-x-1:")
    a=-1
    b=1
    delta=0.06
    Golden_x = golden_section_search(f_a, a, b, delta)
    print("黄金分割法","x=",Golden_x,"f(x)=",f_a(Golden_x))
    Fibonacci_x = fibonacci_search(f_a,a,b,delta)
    print("Fibonacci法","x=",Fibonacci_x,"f(x)=",f_a(Fibonacci_x))
    Binary_x = binary_search(df_a, a, b, delta)
    print("二分法","x=",Binary_x,"f(x)=",f_a(Binary_x))
    Dichotomous_x = dichotomous_search(f_a, a, b, 0.0001, delta)
    print("Dichotomous","x=",Dichotomous_x,"f(x)=",f_a(Dichotomous_x))
    # need to know a proper l.
    # shubert_piyavskii_x = shubert_piyavskii(f_a, a, b, 1)
    # print("Shubert Piyavskii","x=",shubert_piyavskii_x,"f(x)=",shubert_piyavskii_x[1])


    print("f(x)=3x^2-21.6x-1")
    a = 0
    b = 25
    delta = 0.08
    Golden_x = golden_section_search(f_b, a, b, delta)
    print("黄金分割法","x=",Golden_x,"f(x)=",f_a(Golden_x))
    Fibonacci_x = fibonacci_search(f_b, a, b, delta)
    print("Fibonacci法","x=",Fibonacci_x,"f(x)=",f_a(Fibonacci_x))
    Binary_x = binary_search(df_b, a, b, delta)
    print("二分法","x=",Binary_x,"f(x)=",f_a(Binary_x))
    Dichotomous_x = dichotomous_search(f_b, a, b, 0.001, delta)
    print("Dichotomous","x=",Dichotomous_x,"f(x)=",f_a(Dichotomous_x))