## 共轭梯度法
是一种求解线性方程组或者无约束优化问题的迭代方法。它特别适用于大规模稀疏系统。这里，我们应用共轭梯度法来最小化给定的二次函数。

假设我们的目标函数是：

\[ f(x) = x_1 - x_2 + 2x_1^2 + 2x_1x_2 + x_2^2 \]

要最小化这个函数，我们首先需要计算它的梯度。梯度是由偏导数组成的向量，对于上述函数，梯度是：

\[ \nabla f(x) = \left[ \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2} \right] \]

对于函数 \( f(x) \)，梯度是：

\[ \nabla f(x) = \left[ 1 + 4x_1 + 2x_2, -1 + 2x_1 + 2x_2 \right] \]

接下来，我们使用共轭梯度算法的步骤来找到最小值。初始点 \( x^{(0)} = (0, 0)^T \)，容差 \( \epsilon = 10^{-6} \)。

以下是共轭梯度算法的步骤，以及相应的Python代码实现：

1. 初始化： \( x^{(0)} \)，计算 \( r^{(0)} = -\nabla f(x^{(0)}) \)，设置 \( p^{(0)} = r^{(0)} \)， \( k = 0 \)
2. 如果 \( \|r^{(k)}\| < \epsilon \)，停止迭代
3. 计算 \( \alpha_k = \frac{{r^{(k)T} r^{(k)}}}{{p^{(k)T} A p^{(k)}}} \)，其中 \( A \) 是Hessian矩阵
4. 更新解 \( x^{(k+1)} = x^{(k)} + \alpha_k p^{(k)} \)
5. 计算新的残差 \( r^{(k+1)} = r^{(k)} - \alpha_k A p^{(k)} \)
6. 如果 \( \|r^{(k+1)}\| < \epsilon \)，停止迭代
7. 计算 \( \beta_k = \frac{{r^{(k+1)T} r^{(k+1)}}}{{r^{(k)T} r^{(k)}}} \)
8. 更新搜索方向 \( p^{(k+1)} = r^{(k+1)} + \beta_k p^{(k)} \)
9. \( k = k + 1 \)，回到步骤2

Python代码实现：

```python
import numpy as np

# 计算目标函数的值
def f(x):
    return x[0] - x[1] + 2 * x[0]**2 + 2 * x[0] * x[1] + x[1]**2

# 计算梯度的值
def grad_f(x):
    return np.array([1 + 4 * x[0] + 2 * x[1], -1 + 2 * x[0] + 2 * x[1]])

# 共轭梯度法
def conjugate_gradient(x0, epsilon=1e-6):
    # 初始化
    x = x0
    r = -grad_f(x)  # 梯度的负值
    p = r
    k = 0  # 迭代次数

    while np.linalg.norm(r) > epsilon:
        # A是目标函数的Hessian矩阵
        A = np.array([[4, 2], [2, 2]])
        # 计算步长alpha
        alpha = r.T @ r / (p.T @ A @ p)
        # 更新解
        x = x + alpha * p
        # 计算新的残差
        r_new = r - alpha * (A @ p)
        
        if np.linalg.norm(r_new) < epsilon:
            break

        # 更新beta
        beta = r_new.T @ r_new / (r.T @ r)
        # 更新搜索方向
        p = r_new + beta * p
        # 更新残差
        r = r_new
        # 迭代次数增加
        k += 1

    return x, f(x), k

# 初始点
x0 = np.array([0, 0])
# 容差
epsilon = 1e-6

# 使用共轭梯度法求解
solution, function_value, iterations = conjugate_gradient(x0, epsilon)

print("Solution: ", solution)
print("Function value at solution: ", function_value)
print("Number of iterations: ", iterations)
```


### 黄金分割法流程：

黄金分割法是一种寻找一元函数局部最小值的方法。对于单峰函数（在给定区间内只有一个最小值点）表现良好。流程如下：

1. 定义初始区间 \([a, b]\) 和黄金分割比例 \(\varphi = \frac{1 + \sqrt{5}}{2} \approx 1.618\)。
2. 计算两个内点 \(x_1 = b - \frac{b-a}{\varphi}\) 和 \(x_2 = a + \frac{b-a}{\varphi}\)。
3. 比较 \(f(x_1)\) 和 \(f(x_2)\)。
4. 如果 \(f(x_1) > f(x_2)\)，新的区间将是 \([x_1, b]\)，否则将是 \([a, x_2]\)。
5. 重复步骤2到4，直到满足终止条件，比如区间宽度小于给定精度 \(\sigma\)。
6. 最小值大致位于最终区间的中点。

### 黄金分割法代码：

```python
def golden_section_search(f, a, b, sigma):
    # 定义黄金分割比例
    phi = (1 + 5**0.5) / 2

    while (b - a) > sigma:
        # 计算内点
        x1 = b - (b - a) / phi
        x2 = a + (b - a) / phi
        
        # 比较函数值并更新区间
        if f(x1) > f(x2):
            a = x1
        else:
            b = x2

    # 返回区间中点作为最小值点的近似
    return (a + b) / 2

# 定义目标函数
def f(x):
    return 2 * x**2 - x - 1

# 初始区间和区间精度
a0 = -1
b0 = 1
sigma = 0.06

# 使用黄金分割法求解
minimum = golden_section_search(f, a0, b0, sigma)

print("Approximate minimum: ", minimum)
print("Function value at approximate minimum: ", f(minimum))
```


### 斐波那契数列法流程：

斐波那契数列法是一种基于斐波那契数列的搜索技术，用于在有界区间上找到一元函数的最小值。该方法的流程如下：

1. 根据所需的精度确定斐波那契数列长度 \(N\)，使得 \(F_N > (b-a)/\sigma\)，其中 \(F_N\) 是斐波那契数列的第 \(N\) 项，\(a\) 和 \(b\) 是初始搜索区间的端点，\(\sigma\) 是区间精度。
2. 使用斐波那契数列计算两个内点 \(x_1 = a + \frac{F_{N-2}}{F_N}(b-a)\) 和 \(x_2 = a + \frac{F_{N-1}}{F_N}(b-a)\)。
3. 比较 \(f(x_1)\) 和 \(f(x_2)\)。
4. 如果 \(f(x_1) > f(x_2)\)，新的区间将是 \([x_1, b]\)，否则将是 \([a, x_2]\)。
5. 更新 \(N\) 为 \(N-1\)，重新计算 \(x_1\) 和 \(x_2\)。
6. 重复步骤3到5，直到 \(N\) 为1。
7. 最小值大致位于最终区间的中点。

### 斐波那契数列法代码：

```python
def fibonacci_search(f, a, b, sigma):
    # 生成斐波那契数列
    def fibonacci(n):
        fib = [0, 1]
        for i in range(2, n + 1):
            fib.append(fib[-1] + fib[-2])
        return fib
    
    # 计算斐波那契数列长度N
    N = 0
    while fibonacci(N)[-1] < (b - a) / sigma:
        N += 1

    # 获取斐波那契数列
    fib = fibonacci(N)

    # 开始斐波那契搜索
    for i in range(N, 1, -1):
        x1 = a + fib[i-2]/fib[i]*(b - a)
        x2 = a + fib[i-1]/fib[i]*(b - a)

        if f(x1) > f(x2):
            a = x1
        else:
            b = x2

    # 返回区间中点作为最小值点的近似
    return (a + b) / 2

# 定义目标函数
def f(x):
    return 2 * x**2 - x - 1

# 初始区间和区间精度
a0 = -1
b0 = 1
sigma = 0.06

# 使用斐波那契数列法求解
minimum = fibonacci_search(f, a0, b0, sigma)

print("Approximate minimum: ", minimum)
print("Function value at approximate minimum: ", f(minimum))
```

将这段代码复制到Python环境中运行，就能使用斐波那契数列法寻找给出函数的近似最小值点。注意，斐波那契数列法会因为 \(N\) 的值变得非常大而在计算上变得低效，所以它通常适用于精度要求不是特别高的情况。

### 二分法流程：

二分法（又称为二分搜索法）是一种在连续函数上确定最小值的区间缩减方法。对于凸函数或具有单一最小值的函数，二分法特别有效。流程如下：

1. 确定初始搜索区间 \([a, b]\)。
2. 在区间 \([a, b]\) 中选择两个测试点 \(x_1\) 和 \(x_2\)，使得 \(x_1 = \frac{a + b}{2} - \delta\) 和 \(x_2 = \frac{a + b}{2} + \delta\)，其中 \(\delta\) 是一个很小的正数。
3. 比较 \(f(x_1)\) 和 \(f(x_2)\) 的值。
4. 如果 \(f(x_1) < f(x_2)\)，那么新的搜索区间变为 \([a, x_2]\)，否则新的搜索区间变为 \([x_1, b]\)。
5. 重复步骤2到4，直到满足终止条件，比如区间宽度小于给定的精度 \(\sigma\)。
6. 最小值大致位于最终区间的中点。

### 二分法代码：

```python
def bisection_method(f, a, b, sigma):
    while (b - a) / 2 > sigma:
        # 中点
        mid = (a + b) / 2
        # 选择测试点
        delta = sigma / 2
        x1 = mid - delta
        x2 = mid + delta
        
        # 比较函数值
        if f(x1) < f(x2):
            b = x2
        else:
            a = x1

    # 返回区间中点作为最小值点的近似
    return (a + b) / 2

# 定义目标函数
def f(x):
    return 2 * x**2 - x - 1

# 初始区间和区间精度
a0 = -1
b0 = 1
sigma = 0.06

# 使用二分法求解
minimum = bisection_method(f, a0, b0, sigma)

print("Approximate minimum: ", minimum)
print("Function value at approximate minimum: ", f(minimum))
```

将这段代码复制到Python环境中运行，就能使用二分法寻找给出函数的近似最小值点。二分法是一个简单高效的方法，特别适用于一元凸函数的最小化问题。

### Shubert-Piyavskii方法流程：

Shubert-Piyavskii方法是一种求解非光滑一维优化问题的确定性全局优化算法。该方法通过构造目标函数的下界（Lipschitz下界）来确定全局最小值。流程如下：

1. 选择一个Lipschitz常数 \(L\)，它是一个大于或等于目标函数 \(f(x)\) 斜率绝对值的正常数。
2. 从初始区间 \([a, b]\) 开始，选择一组点 \(x_i\) 分布在该区间上。
3. 在每个点 \(x_i\)，根据Lipschitz常数构造势能线（斜率为 \(\pm L\) 的直线），使得这些线位于函数 \(f(x)\) 的下方。
4. 计算这些势能线的交点，找到交点中的最低点 \(x^*\)。
5. 将最低点 \(x^*\) 作为新的搜索点，计算函数值 \(f(x^*)\)。
6. 更新势能线，重复步骤3到5，直至满足收敛准则（例如，区间宽度小于精度 \(\sigma\)）。

注意：选择合适的Lipschitz常数 \(L\) 是该方法的关键。如果 \(L\) 太小，算法可能无法找到全局最小值；如果 \(L\) 太大，算法可能过于保守且收敛速度较慢。



### 总的代码
```python
import numpy as np

# 目标函数


# def f(x):
#     return 2 * x**2 - x - 1
def f(x):
    return 3 * x**2 -21.6 * x -1

# def grad(x):
#     return 4 * x - 1

def grad(x):
    return 6 * x - 21.6
# 黄金分割法


def golden_section_search(f, a, b, sigma):
    phi = (1 + np.sqrt(5)) / 2
    while (b - a) > sigma:
        x1 = b - (b - a) / phi
        x2 = a + (b - a) / phi
        if f(x1) > f(x2):
            a = x1
        else:
            b = x2
    return (a + b) / 2

# 斐波那契数列法


def fibonacci_search(f, a, b, sigma):
    def fibonacci(n):
        fib_seq = [0, 1]
        for i in range(2, n + 1):
            fib_seq.append(fib_seq[-1] + fib_seq[-2])
        return fib_seq
    N = 0
    fib_seq = fibonacci(N)
    while fib_seq[-1] < (b - a) / sigma:
        N += 1
        fib_seq = fibonacci(N)
    for i in range(N, 1, -1):
        x1 = a + fib_seq[i-2]/fib_seq[i]*(b - a)
        x2 = a + fib_seq[i-1]/fib_seq[i]*(b - a)
        if f(x1) > f(x2):
            a = x1
        else:
            b = x2
    return (a + b) / 2

# 二分法


def bisection_method(f, a, b, sigma):
    while (b - a) / 2 > sigma:
        mid = (a + b) / 2
        delta = sigma / 2
        x1 = mid - delta
        x2 = mid + delta
        if f(x1) < f(x2):
            b = x2
        else:
            a = x1
    return (a + b) / 2

# Shubert-Piyavskii方法


def shubert_piyavskii(f, a, b, sigma):
    m = (a+b)/2
    while abs(a-b) > sigma:
        a_d = grad(a)
        b_d = grad(b)
        m = (a+b)/2
        m_d = grad(m)
        x1 = (a * a_d - m * m_d + f(m) - f(a))/(a_d - m_d)
        x2 = (-1 * m * m_d - b * b_d + f(b)-f(m))/(-1 * m_d - b_d)
        if f(x1) < f(x2):
            b = m
        else:
            a = m
    return m


# 初始区间和区间精度
a0 = 0
b0 = 25
sigma = 0.08

# 使用不同方法求解
print("Golden Section Search:")
print("Minimum: ", golden_section_search(f, a0, b0, sigma))

print("\nFibonacci Search:")
print("Minimum: ", fibonacci_search(f, a0, b0, sigma))

print("\nBisection Method:")
print("Minimum: ", bisection_method(f, a0, b0, sigma))

print("\nShubert-Piyavskii Method:")
print("Minimum: ", shubert_piyavskii(f, a0, b0, sigma))

```

### Goldstein方法流程：

Goldstein方法用于在一维搜索中选择步长，确保新的迭代点满足特定的不等式条件。这些条件是为了确保取得足够的函数值下降。流程如下：

1. 给定当前迭代点 \( x \)，搜索方向 \( d \)，以及目标函数 \( f \)。
2. 选择参数 \( c \in (0, 0.5) \)，通常取 \( c = 0.1 \) 或 \( c = 0.2 \)。
3. 选择初始步长 \( \lambda \) 和缩放因子 \( \tau \in (0, 1) \)，通常 \( \tau = 0.5 \)。
4. 计算 \( f(x) \) 和 \( \nabla f(x)^T d \)。
5. 检查 \( f(x + \lambda d) \) 是否满足Goldstein条件：
   \[ f(x) + c \lambda \nabla f(x)^T d \leq f(x + \lambda d) \leq f(x) + (1-c) \lambda \nabla f(x)^T d \]
6. 如果 \( f(x + \lambda d) \) 太大，则减小步长 \( \lambda \)，返回步骤5。
7. 如果 \( f(x + \lambda d) \) 太小，则增加步长 \( \lambda \)，返回步骤5。
8. 如果Goldstein条件满足，则接受 \( \lambda \) 作为搜索步长。

### Goldstein方法代码：

```python
import numpy as np

# 目标函数
def f(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

# 目标函数的梯度
def grad_f(x):
    df_dx1 = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    df_dx2 = 200 * (x[1] - x[0]**2)
    return np.array([df_dx1, df_dx2])

# Goldstein搜索方法
def goldstein_search(f, grad_f, x, d, c=0.2, tau=0.5, lambda_init=1, max_iter=100):
    lambda_ = lambda_init
    phi_0 = f(x)
    phi_prime_0 = grad_f(x).dot(d)
    
    # 初始的Goldstein不等式条件
    def goldstein_conditions(lambda_):
        phi_lambda = f(x + lambda_ * d)
        lower_bound = phi_0 + c * lambda_ * phi_prime_0
        upper_bound = phi_0 + (1 - c) * lambda_ * phi_prime_0
        return phi_lambda >= lower_bound and phi_lambda <= upper_bound
    
    for _ in range(max_iter):
        if goldstein_conditions(lambda_):
            return lambda_
        else:
            # 不满足条件时调整步长
            phi_lambda = f(x + lambda_ * d)
            if phi_lambda < phi_0 + c * lambda_ * phi_prime_0:
                lambda_ /= tau  # 减小步长
            elif phi_lambda > phi_0 + (1 - c) * lambda_ * phi_prime_0:
                lambda_ *= tau  # 增加步长
    return lambda_

# 初始点和搜索方向
x0 = np.array([-1, 1])
d = np.array([1, 1])

# 使用Goldstein方法计算步长
lambda_star = goldstein_search(f, grad_f, x0, d)

print("Selected step size (lambda): ", lambda_star)
```

请将此代码复制到Python环境中运行，它将使用Goldstein搜索方法为给定的目标函数和搜索方向选择一个步长。在这个实现中，我们假设初始步长 `lambda_init` 为1，并通过参数 `c` 和 `tau` 控制步长的增加和减少，直到满足Goldstein条件或达到最大迭代次数。


### 改进的Goldstein方法流程：

改进的Goldstein方法在原始的Goldstein方法基础上增加了步长增加的机制。这样做是为了在保证足够的函数值下降的同时，使算法能够在较大的步长上快速前进。流程如下：

1. 给定当前迭代点 \( x \)，搜索方向 \( d \)，以及目标函数 \( f \)。
2. 选择参数 \( c \in (0, 0.5) \)，通常取 \( c = 0.1 \) 或 \( c = 0.2 \)。
3. 选择初始步长 \( \lambda \) 和缩放因子 \( \tau \in (0, 1) \)，通常 \( \tau = 0.5 \)。
4. 计算 \( f(x) \) 和 \( \nabla f(x)^T d \)。
5. 检查 \( f(x + \lambda d) \) 是否满足改进的Goldstein条件：
   \[ f(x) + c \lambda \nabla f(x)^T d \leq f(x + \lambda d) \leq f(x) + (1-c) \lambda \nabla f(x)^T d \]
6. 如果 \( f(x + \lambda d) \) 太大，则减小步长 \( \lambda \)，返回步骤5。
7. 如果 \( f(x + \lambda d) \) 太小（即满足第一个不等式但不满足第二个不等式），则先增加步长 \( \lambda \) 并检查条件，如果不满足则减小步长，直到找到合适的步长。
8. 如果改进的Goldstein条件满足，则接受 \( \lambda \) 作为搜索步长。

### 改进的Goldstein方法代码：

```python
def goldstein_search(f, grad_f, x, d, c=0.2, tau=0.5, lambda_init=1, max_iter=100):
    lambda_ = lambda_init
    phi_0 = f(x)
    phi_prime_0 = grad_f(x).dot(d)
    
    for _ in range(max_iter):
        phi_lambda = f(x + lambda_ * d)
        # 下界和上界条件
        lower_bound = phi_0 + c * lambda_ * phi_prime_0
        upper_bound = phi_0 + (1 - c) * lambda_ * phi_prime_0
        
        if phi_lambda > upper_bound:  # 如果超出上界，减小步长
            lambda_ *= tau
        elif phi_lambda < lower_bound:  # 如果在下界之内，尝试增加步长
            lambda_prev = lambda_
            lambda_ /= tau
            # 检查增大步长后是否满足条件
            if f(x + lambda_ * d) > upper_bound:
                lambda_ = lambda_prev  # 如果不满足，恢复原来的步长并退出循环
                break
        else:  # 如果在上下界之间，接受当前步长
            break
    
    return lambda_

# 目标函数和梯度的定义及其他参数与前面相同
# ...

# 使用改进的Goldstein方法计算步长
lambda_star = goldstein_search(f, grad_f, x0, d)

print("Selected step size (lambda): ", lambda_star)
```

在这段代码中，当检测到步长太小（即函数值下降太少）时，我们尝试增加步长并再次检查Goldstein条件。如果新的步长导致函数值超出了上界，我们就恢复原来的步长并接受它。

请注意，这是改进的Goldstein方法的一个简化实现，它并不包括所有可能的情况，如何选择最优步长可能需要更多的考虑。在实际应用中，可能需要根据具体问题调整参数和策略。

### Armijo 法流程：

Armijo法，又称为足够下降法则，用于在梯度下降和其他迭代优化算法中选择步长。它确保新的迭代点在目标函数值上有足够的下降。流程如下：

1. 给定当前迭代点 \(x\)，搜索方向 \(d\)（通常为梯度下降方向的负方向），目标函数 \(f\)，以及梯度 \(\nabla f(x)\)。
2. 选择参数 \(c \in (0, 1)\)，通常很小如 \(c = 10^{-4}\)。
3. 选择初始步长 \(\lambda\)，通常为1。
4. 选择步长减少因子 \(\tau \in (0, 1)\)，通常为0.5或0.8。
5. 检查Armijo条件：
   \[ f(x + \lambda d) \leq f(x) + c \lambda \nabla f(x)^T d \]
6. 如果条件不满足，减小步长 \(\lambda := \tau \lambda\)，然后返回步骤5。
7. 如果条件满足，接受 \(\lambda\) 作为步长。

Armijo法保证了每次迭代至少取得了一定比例的预期下降。

### Armijo法代码：

```python
def armijo_line_search(f, grad_f, x, d, c=1e-4, tau=0.5, lambda_init=1):
    lambda_ = lambda_init
    phi_0 = f(x)
    phi_prime_0 = grad_f(x).dot(d)

    while f(x + lambda_ * d) > phi_0 + c * lambda_ * phi_prime_0:
        lambda_ *= tau  # 缩减步长

    return lambda_

# 目标函数
def f(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

# 目标函数的梯度
def grad_f(x):
    df_dx1 = -400 * (x[1] - x[0]**2) * x[0] - 2 * (1 - x[0])
    df_dx2 = 200 * (x[1] - x[0]**2)
    return np.array([df_dx1, df_dx2])

# 初始点和搜索方向
x0 = np.array([-1, 1])
d = -grad_f(x0)  # 在梯度下降中，搜索方向是梯度的负方向

# 使用Armijo法计算步长
lambda_star = armijo_line_search(f, grad_f, x0, d)

print("Selected step size (lambda): ", lambda_star)
```

这段代码使用Armijo法在给定的目标函数和搜索方向上选择一个步长。请注意，本代码实现假设初始步长 `lambda_init` 为1，通常情况下这是一个合理的起点。在实际应用中，你可能需要根据问题的特性调整参数 `c` 和 `tau`。

### Wolfe-Powell 方法流程：

Wolfe-Powell 方法是一种用于在一维搜索中选择步长的技术，该方法结合了Armijo法则（即足够下降条件）和曲率条件。它确保所选步长不仅减少了函数值，而且保证了曲率条件以避免选择太小的步长。流程如下：

1. 给定当前迭代点 \(x\)，搜索方向 \(d\)（通常为梯度下降方向的负方向），目标函数 \(f\)，以及梯度 \(\nabla f(x)\)。
2. 选择参数 \(0 < c_1 < c_2 < 1\)，典型的选择是 \(c_1 = 10^{-4}\)，\(c_2 = 0.9\)。
3. 选择初始步长 \(\lambda\)，通常为1。
4. 选择步长减少因子 \(\tau \in (0, 1)\)，通常为0.5或0.8。
5. 检查Armijo条件（足够下降条件）：
   \[ f(x + \lambda d) \leq f(x) + c_1 \lambda \nabla f(x)^T d \]
6. 检查Wolfe条件（曲率条件）：
   \[ \nabla f(x + \lambda d)^T d \geq c_2 \nabla f(x)^T d \]
7. 如果仅Armijo条件满足，减小步长 \(\lambda := \tau \lambda\)，然后返回步骤5。
8. 如果Armijo条件不满足或者Wolfe条件不满足，调整步长，并返回步骤5。
9. 如果两个条件都满足，接受 \(\lambda\) 作为步长。

### Wolfe-Powell 方法代码：

```python
def wolfe_powell_line_search(f, grad_f, x, d, c1=1e-4, c2=0.9, lambda_init=1, tau=0.5, max_iter=100):
    lambda_ = lambda_init
    phi_0 = f(x)
    phi_prime_0 = grad_f(x).dot(d)
    
    for _ in range(max_iter):
        x_new = x + lambda_ * d
        phi_lambda = f(x_new)
        phi_prime_lambda = grad_f(x_new).dot(d)
        
        # 检查Armijo条件（足够下降）
        if phi_lambda > phi_0 + c1 * lambda_ * phi_prime_0:
            lambda_ *= tau
            continue
        
        # 检查Wolfe条件（曲率）
        if phi_prime_lambda < c2 * phi_prime_0:
            lambda_ *= tau
            continue
        
        # 如果两个条件都满足，则接受当前步长
        return lambda_
    
    return lambda_

# 目标函数和梯度的定义与前面相同
# ...

# 初始点和搜索方向
x0 = np.array([-1, 1])
d = -grad_f(x0)  # 在梯度下降中，搜索方向是梯度的负方向

# 使用Wolfe-Powell方法计算步长
lambda_star = wolfe_powell_line_search(f, grad_f, x0, d)

print("Selected step size (lambda): ", lambda_star)
```

该代码实现了Wolfe-Powell方法，用于在给定的目标函数和搜索方向上选择一个步长。在这个算法中，我们需要同时检查足够下降条件（Armijo条件）和曲率条件（Wolfe条件），以确保所选的步长既可以减小目标函数值，又不会因为太小的步长而导致算法进展缓慢。

请注意，这个实现是一个基本的框架，可能需要根据具体问题调整参数 `c1`、`c2` 和 `tau`，以及增加额外的逻辑以处理可能的边界情况。

### WP改进规则（强Wolfe条件）流程：

强Wolfe条件是在Wolfe条件的基础上添加了另一个限制来避免过小的步长。除了满足Armijo（足够下降）条件和曲率条件外，还要求梯度在搜索方向上的绝对值不得大于曲率条件的梯度乘以一个负的常数。流程如下：

1. 给定当前迭代点 \(x\)，搜索方向 \(d\)，目标函数 \(f\)，以及梯度 \(\nabla f(x)\)。
2. 选择参数 \(0 < c_1 < c_2 < 1\)，典型的选择是 \(c_1 = 10^{-4}\)，\(c_2 = 0.9\)。
3. 选择初始步长 \(\lambda\)，通常为1。
4. 选择步长缩减因子 \(\tau \in (0, 1)\)，通常为0.5或0.8。
5. 检查Armijo条件（足够下降条件）：
   \[ f(x + \lambda d) \leq f(x) + c_1 \lambda \nabla f(x)^T d \]
6. 检查强Wolfe条件（曲率条件）：
   \[ |\nabla f(x + \lambda d)^T d| \leq c_2 |\nabla f(x)^T d| \]
7. 如果仅Armijo条件满足，减小步长 \(\lambda := \tau \lambda\)，然后返回步骤5。
8. 如果Armijo条件不满足或者强Wolfe条件不满足，调整步长，并返回步骤5。
9. 如果两个条件都满足，接受 \(\lambda\) 作为步长。

### WP改进规则代码：

```python
def strong_wolfe_line_search(f, grad_f, x, d, c1=1e-4, c2=0.9, lambda_init=1, tau=0.5, max_iter=100):
    lambda_ = lambda_init
    phi_0 = f(x)
    phi_prime_0 = grad_f(x).dot(d)
    
    for _ in range(max_iter):
        x_new = x + lambda_ * d
        phi_lambda = f(x_new)
        phi_prime_lambda = grad_f(x_new).dot(d)
        
        # 检查Armijo条件（足够下降）
        if phi_lambda > phi_0 + c1 * lambda_ * phi_prime_0:
            lambda_ *= tau
            continue
        
        # 检查强Wolfe条件（曲率）
        if abs(phi_prime_lambda) > c2 * abs(phi_prime_0):
            lambda_ *= tau
            continue
        
        # 如果两个条件都满足，则接受当前步长
        return lambda_
    
    return lambda_

# 目标函数和梯度的定义与前面相同
# ...

# 初始点和搜索方向
x0 = np.array([-1, 1])
d = -grad_f(x0)  # 在梯度下降中，搜索方向是梯度的负方向

# 使用强Wolfe条件计算步长
lambda_star = strong_wolfe_line_search(f, grad_f, x0, d)

print("Selected step size (lambda): ", lambda_star)
```

这段代码实现了强Wolfe条件的搜索，用于在给定的目标函数和搜索方向上选择一个步长。在实现中，我们使用了两个参数 `c1` 和 `c2` 来检查Armijo条件和强Wolfe条件，并使用一个缩减因子 `tau` 来调整步长。如果强Wolfe条件不满足，我们缩减步长并重新检查两个条件。

请注意，这个实现是一个基本的框架，可能需要根据具体问题调整参数 `c1`、`c2` 和 `tau`，以及增加额外的逻辑以处理可能的边界情况。

### 总的代码

```python
import numpy as np

# 目标函数
def f(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

# 目标函数的梯度
def grad_f(x):
    df_dx1 = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    df_dx2 = 200 * (x[1] - x[0]**2)
    return np.array([df_dx1, df_dx2])

# Goldstein搜索方法
def goldstein_search(f, grad_f, x, d, c=0.2, tau=0.5, lambda_init=1, max_iter=100):
    # ... (之前给出的Goldstein方法代码)
    # ...

# 改进的Goldstein搜索方法
def improved_goldstein_search(f, grad_f, x, d, c=0.2, tau=0.5, lambda_init=1, max_iter=100):
    # ... (之前给出的改进的Goldstein方法代码)
    # ...

# Armijo搜索方法
def armijo_line_search(f, grad_f, x, d, c=1e-4, tau=0.5, lambda_init=1):
    # ... (之前给出的Armijo法代码)
    # ...

# Wolfe-Powell搜索方法
def wolfe_powell_line_search(f, grad_f, x, d, c1=1e-4, c2=0.9, lambda_init=1, tau=0.5, max_iter=100):
    # ... (之前给出的Wolfe-Powell方法代码)
    # ...

# 强Wolfe条件搜索方法
def strong_wolfe_line_search(f, grad_f, x, d, c1=1e-4, c2=0.9, lambda_init=1, tau=0.5, max_iter=100):
    # ... (之前给出的强Wolfe条件方法代码)
    # ...

# 初始点和搜索方向
x0 = np.array([-1, 1])
d = np.array([1, 1])  # 给定的搜索方向

# 使用不同的方法计算步长，并打印极小值和极小值点
methods = {
    "Goldstein": goldstein_search,
    "Improved Goldstein": improved_goldstein_search,
    "Armijo": armijo_line_search,
    "Wolfe-Powell": wolfe_powell_line_search,
    "Strong Wolfe": strong_wolfe_line_search
}

for name, method in methods.items():
    lambda_star = method(f, grad_f, x0, d)
    x_min = x0 + lambda_star * d
    f_min = f(x_min)
    print(f"{name} Method:")
    print(f"Selected step size (lambda): {lambda_star}")
    print(f"Minimum point: {x_min}")
    print(f"Function value at minimum point: {f_min}\n")
```

### DFP
DFP（Davidon-Fletcher-Powell）方法是一种拟牛顿法，用于求解无约束优化问题。以下是DFP算法的基本流程：

1. **初始化参数**：
   - 设置初始点 \( x^{(0)} \)。
   - 设置初始Hessian矩阵的逆近似 \( H^{(0)} \) 为单位矩阵。
   - 设置收敛阈值 \(\epsilon\)。

2. **迭代**：
   对 \( k = 0, 1, 2, \ldots \) 执行以下步骤，直到 \( \|\nabla f(x^{(k)})\| < \epsilon \)：

   a. **计算搜索方向**：
      - 计算搜索方向 \( d^{(k)} = -H^{(k)} \nabla f(x^{(k)}) \)。

   b. **线搜索**：
      - 在方向 \( d^{(k)} \) 上进行精确一维搜索，找到使得 \( f(x^{(k)} + \alpha d^{(k)}) \) 最小的 \( \alpha_k \)。

   c. **更新迭代点**：
      - 更新迭代点 \( x^{(k+1)} = x^{(k)} + \alpha_k d^{(k)} \)。

   d. **计算更新量**：
      - 计算 \( s^{(k)} = x^{(k+1)} - x^{(k)} \) 和 \( y^{(k)} = \nabla f(x^{(k+1)}) - \nabla f(x^{(k)}) \)。

   e. **更新Hessian的逆近似**：
      - 计算 \( H^{(k+1)} \) 使用DFP更新公式：
        \[
        H^{(k+1)} = H^{(k)} + \frac{s^{(k)} (s^{(k)})^T}{(s^{(k)})^T y^{(k)}} - \frac{H^{(k)} y^{(k)} (y^{(k)})^T H^{(k)}}{(y^{(k)})^T H^{(k)} y^{(k)}}
        \]

3. **检查收敛性**：
   - 如果 \( \|\nabla f(x^{(k+1)})\| < \epsilon \)，则停止迭代；否则，返回步骤2。

Code:
```python
import numpy as np

# 目标函数
def f(x):
    return 10 * x[0]**2 + x[1]**2

# 目标函数的梯度
def grad_f(x):
    return np.array([20 * x[0], 2 * x[1]])

# 精确一维搜索，这里使用了简单的黄金分割搜索作为示例
def line_search(f, x, d):
    a, b = 0, 1
    tau = (np.sqrt(5) - 1) / 2
    alpha1 = a + (1 - tau) * (b - a)
    alpha2 = a + tau * (b - a)
    while (b - a) > 1e-5:
        if f(x + alpha1 * d) < f(x + alpha2 * d):
            b = alpha2
        else:
            a = alpha1
        alpha1 = a + (1 - tau) * (b - a)
        alpha2 = a + tau * (b - a)
    return (alpha1 + alpha2) / 2

# DFP方法
def dfp(f, grad_f, x0, epsilon=1e-6, max_iter=100):
    x = x0
    H = np.eye(len(x))  # 初始化H矩阵为单位矩阵
    for _ in range(max_iter):
        grad = grad_f(x)
        if np.linalg.norm(grad) < epsilon:
            break
        d = -H.dot(grad)  # 计算搜索方向
        alpha = line_search(f, x, d)  # 精确一维搜索
        s = alpha * d
        x_new = x + s  # 更新迭代点
        y = grad_f(x_new) - grad
        rho = 1.0 / (y.dot(s))
        Hys = np.outer(H.dot(y), s)
        # 更新H矩阵
        H = H + rho * np.outer(s, s) - (Hys + Hys.T) / (y.dot(s)) + rho**2 * s.dot(H.dot(y)) * np.outer(s, s)
        x = x_new
    return x, f(x)

# 初始点
x0 = np.array([0.1, 1])

# 使用DFP方法求解
minimum_point, minimum_value = dfp(f, grad_f, x0)

print("Minimum value: ", minimum_value)
print("Minimum point: ", minimum_point)
```
### BFGS
BFGS（Broyden-Fletcher-Goldfarb-Shanno）方法是拟牛顿法的一种，用于求解无约束优化问题。它通过构建目标函数的Hessian矩阵的逆近似来寻找最小值。以下是BFGS算法的基本流程：

1. **初始化参数**：
   - 设置初始点 \( x^{(0)} \)。
   - 设置初始Hessian矩阵的逆近似 \( H^{(0)} \) 为单位矩阵。
   - 设置收敛阈值 \(\epsilon\)。

2. **迭代过程**：
   对 \( k = 0, 1, 2, \ldots \) 执行以下步骤：

   a. **计算搜索方向**：
      - 使用当前的Hessian矩阵的逆近似 \( H^{(k)} \) 计算搜索方向 \( p^{(k)} = -H^{(k)} \nabla f(x^{(k)}) \)。

   b. **线搜索**：
      - 在方向 \( p^{(k)} \) 上进行精确一维搜索，找到步长 \(\alpha_k\) 使得 \( f(x^{(k)} + \alpha_k p^{(k)}) \) 尽可能小。

   c. **更新迭代点**：
      - 更新点 \( x^{(k+1)} = x^{(k)} + \alpha_k p^{(k)} \)。

   d. **计算梯度和步长差**：
      - 计算梯度差 \( y^{(k)} = \nabla f(x^{(k+1)}) - \nabla f(x^{(k)}) \)。
      - 计算步长差 \( s^{(k)} = x^{(k+1)} - x^{(k)} \)。

   e. **更新Hessian矩阵的逆近似**：
      - 计算标量值 \( \rho^{(k)} = 1 / (y^{(k)T} s^{(k)}) \)。
      - 使用BFGS更新公式更新Hessian矩阵的逆近似：
        \[
        H^{(k+1)} = \left(I - \rho^{(k)} s^{(k)} y^{(k)T}\right) H^{(k)} \left(I - \rho^{(k)} y^{(k)} s^{(k)T}\right) + \rho^{(k)} s^{(k)} s^{(k)T}
        \]
        其中 \( I \) 是单位矩阵。

   f. **检查收敛性**：
      - 如果 \( \|\nabla f(x^{(k+1)})\| < \epsilon \)，则算法收敛，停止迭代。

3. **输出结果**：
   - 返回最终迭代点 \( x^{(k+1)} \) 作为解，以及 \( f(x^{(k+1)}) \) 作为最小值。

BFGS方法是一种有效的大规模无约束优化算法，它不需要计算Hessian矩阵，而是逐步构建其逆近似，使得每次迭代都朝着更接近最小值的方向前进。

Code：
```python
import numpy as np

# 目标函数
def f(x):
    return x[0]**2 + 4 * x[1]**2 - 4 * x[0] - 8 * x[1]

# 目标函数的梯度
def grad_f(x):
    return np.array([2 * x[0] - 4, 8 * x[1] - 8])

# 精确一维搜索，这里使用了黄金分割搜索作为示例
def line_search(f, x, d):
    a, b = 0, 1
    tau = (np.sqrt(5) - 1) / 2
    alpha1 = a + (1 - tau) * (b - a)
    alpha2 = a + tau * (b - a)
    while (b - a) > 1e-5:
        if f(x + alpha1 * d) < f(x + alpha2 * d):
            b = alpha2
        else:
            a = alpha1
        alpha1 = a + (1 - tau) * (b - a)
        alpha2 = a + tau * (b - a)
    return (alpha1 + alpha2) / 2

# BFGS方法
def bfgs(f, grad_f, x0, epsilon=1e-6, max_iter=1000):
    x = x0
    H = np.eye(len(x))  # 初始化H矩阵为单位矩阵
    I = np.eye(len(x))  # 单位矩阵

    for _ in range(max_iter):
        grad = grad_f(x)
        if np.linalg.norm(grad) < epsilon:
            break
        p = -H.dot(grad)  # 计算搜索方向
        alpha = line_search(f, x, p)  # 精确一维搜索
        s = alpha * p
        x_new = x + s  # 更新迭代点
        y = grad_f(x_new) - grad
        rho = 1.0 / y.dot(s)
        # 计算外积项
        ys = np.outer(y, s)
        sy = np.outer(s, y)
        # BFGS更新公式
        H = (I - rho * ys).dot(H).dot(I - rho * sy) + rho * np.outer(s, s)
        x = x_new

    return x, f(x)

# 初始点
x0 = np.array([0.0, 0.0])

# 使用BFGS方法求解
minimum_point, minimum_value = bfgs(f, grad_f, x0)

print("Minimum value: ", minimum_value)
print("Minimum point: ", minimum_point)
```


二次规划问题通常表示为以下形式：

\[
\min_{x \in \mathbb{R}^n} f(x) = \frac{1}{2} x^T Q x + b^T x
\]

其中$Q \in \mathbb{R}^{n \times n}$是对称正定矩阵（或者至少是半正定矩阵），$b \in \mathbb{R}^n$是常数向量。二次规划问题的特点在于其梯度和Hessian矩阵都是容易计算的：

\[
\nabla f(x) = Qx + b
\]
\[
\nabla^2 f(x) = Q
\]

因为$Q$是正定的，$f(x)$是一个凸函数。接下来我们分析梯度下降法在这种情况下的收敛性。

### 梯度下降法的收敛性分析

梯度下降法的迭代更新规则为：

\[
x_{k+1} = x_k - \alpha_k \nabla f(x_k)
\]

其中$x_k$是第$k$次迭代的点，$\alpha_k$是第$k$次迭代的步长，$\nabla f(x_k)$是$x_k$处的梯度。

为了分析收敛性，我们假设以下条件：

1. **步长**：$\alpha_k$采用固定步长或者通过某种线搜索策略来确定，例如Armijo规则。

2. **正定性**：$Q$是正定矩阵，这意味着对于所有$x \neq 0$都有$x^T Q x > 0$。这确保了$f(x)$是严格凸的，并且有一个唯一的全局最小值点。

### 收敛性证明

考虑一个光滑的凸目标函数$f: \mathbb{R}^n \to \mathbb{R}$，梯度下降法的更新规则为：

\[
x_{k+1} = x_k - \alpha_k \nabla f(x_k)
\]

其中$x_k$是第$k$次迭代的点，$\alpha_k$是第$k$次迭代的步长，$\nabla f(x_k)$是$x_k$处的梯度。

为了证明梯度下降法的收敛性，我们需要几个假设：

1. **凸性**：$f$是凸函数，即对于所有$x,y \in \mathbb{R}^n$和$\theta \in [0, 1]$，有：
   \[
   f(\theta x + (1 - \theta) y) \leq \theta f(x) + (1 - \theta) f(y)
   \]

2. **Lipschitz连续梯度**：$f$的梯度是$L$-Lipschitz连续的，即存在常数$L > 0$，使得对于所有$x,y \in \mathbb{R}^n$，有：
   \[
   \|\nabla f(x) - \nabla f(y)\| \leq L \|x - y\|
   \]
   
3. **步长选择**：步长$\alpha_k$满足$0 < \alpha_k < \frac{2}{L}$。

**证明**：

由于$f$的梯度是$L$-Lipschitz连续的，我们有：

\[
f(x_{k+1}) \leq f(x_k) + \langle \nabla f(x_k), x_{k+1} - x_k \rangle + \frac{L}{2}\|x_{k+1} - x_k\|^2
\]

将$x_{k+1} = x_k - \alpha_k \nabla f(x_k)$代入上式得：

\[
f(x_{k+1}) \leq f(x_k) - \alpha_k \|\nabla f(x_k)\|^2 + \frac{L \alpha_k^2}{2} \|\nabla f(x_k)\|^2
\]

简化得：

\[
f(x_{k+1}) - f(x_k) \leq -\alpha_k \left(1 - \frac{L \alpha_k}{2}\right) \|\nabla f(x_k)\|^2
\]

由于$0 < \alpha_k < \frac{2}{L}$，我们可以保证$1 - \frac{L \alpha_k}{2} > 0$。这意味着$f(x_{k+1}) < f(x_k)$，除非$\nabla f(x_k) = 0$，即$x_k$已经是一点极小值。

因此，只要梯度不为零，目标函数值将持续减少。由于$f$是凸函数，这保证了序列$\{f(x_k)\}$将收敛到$f$的全局最小值。

此外，由于目标函数值的连续减少，我们可以得到梯度序列$\{\nabla f(x_k)\}$必须趋向于零，因为如果存在一个下界$\epsilon > 0$使得对于所有$k$，$\|\nabla f(x_k)\| > \epsilon$，那么函数值将无限减小，这与函数值有下界矛盾。

综上，我们证明了在上述假设下，梯度下降法生成的序列$\{x_k\}$在目标函数值上是单调递减的，并且梯度序列$\{\nabla f(x_k)\}$趋向于零，这意味着该序列收敛到目标函数的极小值点。

牛顿法是一种基于迭代的优化算法，它使用目标函数的一阶和二阶导数信息来寻找极值点。这里我们考虑无约束优化问题，目标函数$f: \mathbb{R}^n \to \mathbb{R}$是二次可微的。牛顿法的迭代更新规则为：

\[
x_{k+1} = x_k - [\nabla^2 f(x_k)]^{-1} \nabla f(x_k)
\]

其中$x_k$是第$k$次迭代的点，$\nabla f(x_k)$是$x_k$处的梯度，$\nabla^2 f(x_k)$是$x_k$处的Hessian矩阵。

## 牛顿法
为了证明牛顿法的局部二次收敛性，我们需要几个假设：

1. **目标函数**：$f$在某个开集$S \subseteq \mathbb{R}^n$上是二次可微的，并且有一个唯一的极小点$x^*$。

2. **Hessian 正定性**：在$x^*$附近，$f$的Hessian矩阵$\nabla^2 f(x)$是正定的，即对于所有$x$接近$x^*$，所有非零向量$z \in \mathbb{R}^n$，都有$z^T \nabla^2 f(x) z > 0$。

3. **初始点**：初始点$x_0$足够接近$x^*$，即$x_0 \in S$。

4. **Hessian Lipschitz连续**：$f$的Hessian在$S$上是$L$-Lipschitz连续的，即存在常数$L > 0$，使得对于所有$x,y \in S$，有：
   \[
   \|\nabla^2 f(x) - \nabla^2 f(y)\| \leq L \|x - y\|
   \]

**证明局部二次收敛性**：

考虑泰勒展开$f$在$x^*$的附近：

\[
f(x) = f(x^*) + \nabla f(x^*)^T (x - x^*) + \frac{1}{2} (x - x^*)^T \nabla^2 f(x^*) (x - x^*) + O(\|x - x^*\|^3)
\]

因为$x^*$是极小点，我们有$\nabla f(x^*) = 0$。由于Hessian矩阵在$x^*$是正定的，我们可以忽略高阶项，得到：

\[
\nabla f(x) \approx \nabla^2 f(x^*) (x - x^*)
\]

牛顿法的更新可以写为：

\[
x_{k+1} = x_k - [\nabla^2 f(x_k)]^{-1} \nabla f(x_k) \approx x_k - [\nabla^2 f(x^*)]^{-1} \nabla^2 f(x^*) (x_k - x^*)
\]

这意味着：

\[
x_{k+1} - x^* \approx (I - [\nabla^2 f(x^*)]^{-1} \nabla^2 f(x^*)) (x_k - x^*)
\]

由于$I - [\nabla^2 f(x^*)]^{-1} \nabla^2 f(x^*) = 0$，我们可以得到$x_{k+1} \approx x^*$，即$x_{k+1}$比$x_k$更接近$x^*$。这表明牛顿法的迭代是在局部二次收敛到$x^*$的。

以上证明是局部收敛性的直观解释，真正的二次收敛性证明涉及到对迭代误差$e_k = x_k - x^*$进行递推关系的分析，并且证明$e_{k+1}\[
e_{k+1} = O(\|e_k\|^2)
\]

这表示当迭代点$x_k$足够接近极小点$x^*$时，误差$e_k$的减少速度至少是其平方。以下是更为详细的证明过程。

**详细证明**：

我们定义迭代误差为$e_k = x_k - x^*$。在$x^*$处进行泰勒展开，我们有：

\[
\nabla f(x_k) = \nabla f(x^*) + \nabla^2 f(x^*)(x_k - x^*) + O(\|x_k - x^*\|^2)
\]

由于$x^*$是极小点，$\nabla f(x^*) = 0$，上式可以简化为：

\[
\nabla f(x_k) = \nabla^2 f(x^*) e_k + O(\|e_k\|^2)
\]

牛顿法的更新规则是：

\[
x_{k+1} = x_k - [\nabla^2 f(x_k)]^{-1} \nabla f(x_k)
\]

将梯度的泰勒展开代入更新规则，我们得到：

\[
x_{k+1} = x_k - [\nabla^2 f(x_k)]^{-1} (\nabla^2 f(x^*) e_k + O(\|e_k\|^2))
\]

为了分析收敛性，我们需要考虑$[\nabla^2 f(x_k)]^{-1}$与$[\nabla^2 f(x^*)]^{-1}$之间的关系。由于Hessian是$L$-Lipschitz连续的，我们有：

\[
\|[\nabla^2 f(x_k)]^{-1} - [\nabla^2 f(x^*)]^{-1}\| \leq \|[\nabla^2 f(x_k)]^{-1}\| \cdot \|[\nabla^2 f(x^*)]^{-1}\| \cdot L \|e_k\|
\]

假设存在$M > 0$使得$\|[\nabla^2 f(x)]^{-1}\| \leq M$对于所有$x$接近$x^*$都成立，那么我们有：

\[
\|[\nabla^2 f(x_k)]^{-1} - [\nabla^2 f(x^*)]^{-1}\| = O(\|e_k\|)
\]

因此，我们可以写出：

\[
x_{k+1} - x^* = e_{k+1} = (I - [\nabla^2 f(x_k)]^{-1} \nabla^2 f(x^*)) e_k + O(\|e_k\|^2)
\]

由于在$x^*$附近，$I - [\nabla^2 f(x_k)]^{-1} \nabla^2 f(x^*)$的项是$O(\|e_k\|)$，我们最终得到：

\[
e_{k+1} = O(\|e_k\|^2)
\]

这表明当我们足够靠近极小点$x^*$时，牛顿法的每次迭代都会使得误差的大小至少按照其当前值的平方来减少.
对于二次规划问题，我们可以直接计算出最优步长$\alpha_k^*$：

\[
\alpha_k^* = \frac{\nabla f(x_k)^T \nabla f(x_k)}{\nabla f(x_k)^T Q \nabla f(x_k)}
\]

这个步长是精确线搜索的结果，它最小化了沿着梯度方向的一维函数。

使用了这个最优步长，我们可以写出梯度的显式形式：

\[
\nabla f(x_{k+1}) = Qx_{k+1} + b = Q(x_k - \alpha_k^* \nabla f(x_k)) + b
\]

简化后得到：

\[
\nabla f(x_{k+1}) = \nabla f(x_k) - \alpha_k^* Q \nabla f(x_k)
\]

将$\alpha_k^*$的定义代入，我们得到：

\[
\nabla f(x_{k+1}) = \left(I - \frac{Q \nabla f(x_k) \nabla f(x_k)^T}{\nabla f(x_k)^T Q \nabla f(x_k)}\right) \nabla f(x_k)
\]

由于$Q$是正定的，我们可以保证$\nabla f(x_k)^T Q \nabla f(x_k) > 0$，并且：

\[
\| \nabla f(x_{k+1}) \| < \| \nabla f(x_k) \|
\]

这意味着梯度的模长在每一次迭代后都会减小。因为$Q$是正定的，我们还可以保证：

\[
f(x_{k+1}) < f(x_k)
\]

这表明函数值在每一次迭代中都严格下降，除非$x_k$已经是最小值点，即$\nabla f(x_k) = 0$。

因此，使用精确线搜索的梯度下降法在二次规划问题上是全局收敛的。如果步长