import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt

# Use the correct normalization for CIFAR-10
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True)

# Move the model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet18().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Function to calculate parameter differences
def calculate_parameter_diff(model, prev_params):
    diff_list = []
    with torch.no_grad():
        for param, prev_param in zip(model.parameters(), prev_params):
            diff_list.append(torch.norm(param - prev_param).item())
    return diff_list

parameter_diffs = []
prev_params = [param.clone() for param in model.parameters()]

num_iterations = 10
for i, data in enumerate(trainloader, 0):
    if i >= num_iterations:
        break

    inputs, labels = data[0].to(device), data[1].to(device)
    y_pred = model(inputs)
    # Loss calculation
    loss = criterion(y_pred, labels)
    # Gradient computation
    optimizer.zero_grad()
    loss.backward()
    # Parameter update
    optimizer.step()

    # Calculate parameter differences
    parameter_diffs.append(calculate_parameter_diff(model, prev_params))
    # Update previous parameters
    prev_params = [param.clone() for param in model.parameters()]

# Plotting the parameter difference distribution
parameter_diffs = np.array(parameter_diffs)
parameter_diffs_flat = parameter_diffs.flatten()
plt.hist(parameter_diffs_flat, bins=100)
plt.title("ResNet18 parameter difference distribution")
plt.show()