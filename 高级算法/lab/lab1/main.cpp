#include "lab1.h"
#include <chrono>

int main()
{
    // 构建示例数据集
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::unordered_set<int>> documents;
    int num_max = 0;
    num_max = reader(documents, "D:\\courses\\Course-Note\\AdvanceAlgorithm\\lab\\lab1\\10k.txt");
    num_max += 10;
    auto time_read = std::chrono::high_resolution_clock::now();
    std::cout << "read time comsume:" << std::chrono::duration_cast<std::chrono::seconds>(time_read - start).count() << std::endl;
    for (int i = 40; i <= 100; i += 10)
    {
        run_experiment(documents, i, 0.5, num_max);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    std::cout << "tot_time"
              << " :" << duration.count() << " seconds" << std::endl;
    return 0;
}