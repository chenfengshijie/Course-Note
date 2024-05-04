
import torch

def f(x):
    return x[0] - x[1] + 2*x[0]**2 + 2*x[0]*x[1] + x[1]**2

def grad_f(x):
    return torch.tensor([4*x[0] + 2*x[1] + 1, 2*x[0] + 4*x[1] - 1])

def step_func(optimizer, x, num_iterations):
    for _ in range(num_iterations):
        optimizer.zero_grad()
        loss = f(x)
        loss.backward()
        optimizer.step()
    return x.detach()

if __name__ == "__main__":
    lr = 0.1
    num_iterations = 200

    optimizers = {
        "SGD": torch.optim.SGD,
        "Momentum": lambda params,lr: torch.optim.SGD(params, lr, momentum=0.9),
        "Adagrad": torch.optim.Adagrad,
        "RMSprop": torch.optim.RMSprop,
        "Adam": torch.optim.Adam
    }

    for name, opt_constructor in optimizers.items():
        x_init = torch.tensor([1.0, 1.0], requires_grad=True)
        optimizer = opt_constructor([x_init], lr=lr)
        x_opt = step_func(optimizer, x_init, num_iterations)
        print(f"{name}:", x_opt.numpy())
        print(f"F(x) = {f(x_opt).item()}")