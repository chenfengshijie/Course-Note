import matplotlib.pyplot as plt

# 定义数据集名称
datasets = ["AOL", "Booking", "100k"]

# 为每个数据集初始化字典
data = {name: {"num_of_hashfunc": [], "precision": [], "recall": [], "time": None} for name in datasets}
# 读取数据
with open("experiment.txt", "r") as f:
    lines = f.readlines()

current_dataset = None
for line in lines:
    line = line.strip()
    if line is None:
        continue

    if line in datasets:  # 确定当前数据集
        current_dataset = line
    elif line.startswith("num_of_hashfunc"):  # 读取hash函数数量
        num = line.split(":")[1]
        data[current_dataset]["num_of_hashfunc"].append(int(num))
    elif line.startswith("precision"):  # 读取精度
        precision = line.split(":")[1]
        data[current_dataset]["precision"].append(float(precision))
    elif line.startswith("recall"):  # 读取召回率
        recall = line.split(":")[1]
        data[current_dataset]["recall"].append(float(recall))
    elif line.startswith("time"):  # 读取时间
        time = line.split(":")[1].rstrip('s')
        data[current_dataset]["time"] = float(time)

# 绘图
for i, dataset in enumerate(datasets):

    plt.subplot(2, 2, i + 1)
    plt.plot(data[dataset]["num_of_hashfunc"], data[dataset]["precision"], marker='o', label='Precision')
    plt.plot(data[dataset]["num_of_hashfunc"], data[dataset]["recall"], marker='o', label='Recall')

    plt.title(f"{dataset} Dataset")
    plt.xlabel("Number of Hash Functions")
    plt.ylabel("Value")
    plt.legend()

plt.tight_layout()
plt.show()
