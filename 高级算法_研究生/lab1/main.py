import random
import time
import matplotlib.pyplot as plt
from enumerate_ import brute_force
from gram_scan import graham_scan
from divide_conquer import divide_conquer
import copy


# Function to generate random points within a square area
def generate_points(num_points, square_size):
    points = []
    for _ in range(num_points):
        x = random.uniform(0, square_size)
        y = random.uniform(0, square_size)
        points.append((x, y))
    return points


# Function to check if the convex hulls produced by all algorithms are consistent
def check_consistency():
    points = generate_points(100, 100)
    res1 = divide_conquer(points[:])
    res2 = graham_scan(points=points[:])
    res3 = brute_force(points=points[:])
    res2.sort(key=lambda x: x[0])
    res1.sort(key=lambda x: x[0])
    res3.sort(key=lambda x: x[0])

    for k1, k2, k3 in zip(res1, res2, res3):
        if not (k1 == k2 == k3):
            print(k1, k2, k3)


# Function to test the performance of the three algorithms and plot the results
def main():
    number_of_points = [i * 1000 for i in range(1, 6)]
    brute_force_times = []
    graham_scan_times = []
    divide_conquer_times = []

    for n in number_of_points:
        points = generate_points(n, 100)

        start_time = time.time()
        brute_force(point=points[:])
        brute_force_times.append(time.time() - start_time)
        print(f"Brute Force: {n} points:time: {time.time() - start_time}")
        start_time = time.time()
        graham_scan(points=points[:])
        graham_scan_times.append(time.time() - start_time)
        print(f"Graham Scan: {n} points:time: {time.time() - start_time}")
        start_time = time.time()
        divide_conquer(point=points[:])
        divide_conquer_times.append(time.time() - start_time)
        print(f"Divide and Conquer: {n} points:time: {time.time() - start_time}")

    plt.figure(figsize=(10, 5))
    plt.plot(number_of_points, brute_force_times, label="Brute Force", marker="o")
    plt.plot(number_of_points, graham_scan_times, label="Graham Scan", marker="s")
    plt.plot(
        number_of_points, divide_conquer_times, label="Divide and Conquer", marker="^"
    )
    plt.xlabel("Number of Points")
    plt.ylabel("Time (seconds)")
    plt.title("Performance Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("convex_hull_algorithms_performance.png")
    plt.show()


if __name__ == "__main__":
    main()
