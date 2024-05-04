import numpy as np

def f(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def df(x):
    return np.array([400 * x[0]**3 - 400 * x[0] * x[1] + 2 * x[0] - 2, 200 * x[1] - 200 * x[0]**2])

def line_search(f, df, x, d, method='Goldstein', a=1.5, b=0.5, rho=0.1, sigma=0.5, lamda=1):
    while True:
        x_n = x + lamda * d
        f_x_n = f(x_n)
        f_x = f(x)
        df_x = df(x)
        dot_product = np.dot(df_x, d)
        
        if f_x_n - f_x > rho * dot_product * lamda:
            lamda *= b
        elif method == 'Goldstein' and f_x_n - f_x < (1 - rho) * dot_product * lamda:
            lamda *= a
        elif method == 'Goldstein-Price' and f_x_n - f_x < sigma * dot_product * lamda:
            lamda *= a
        elif method == 'Wolfe-Powell' and np.dot(df(x_n), d) < sigma * dot_product:
            lamda *= a
        else:
            return lamda, f_x_n

if __name__ == "__main__":
    x = np.array([-1, 1])
    d = np.array([1, 1])
    
    lamda, val = line_search(f, df, x, d, method='Goldstein')
    print("Goldstein法:")
    print("λ=", lamda)
    print("f(x)=", val)
    
    lamda, val = line_search(f, df, x, d, method='Goldstein-Price')
    print("Goldstein-Price法:")
    print("λ=", lamda)
    print("f(x)=", val)
    
    lamda, val = line_search(f, df, x, d, method='Wolfe-Powell')
    print("Wolfe-Powell法:")
    print("λ=", lamda)
    print("f(x)=", val)