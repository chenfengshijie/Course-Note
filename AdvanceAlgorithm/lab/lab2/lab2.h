#ifndef __LAB2_H__
#define __LAB2_H__
#include <vector>
#include <string>
#include <unordered_map>
// 第一个实验所需函数
std::vector<double> generateUniformDistribution(int n, double minVal, double maxVal);
std::vector<double> generateNormalDistribution(int n, double mean, double stddev);
std::vector<double> generateZipfDistribution(int n, double alpha, int numItems);
double select_after_sort(std::vector<double> &nums, int k);
double linear_select(std::vector<double> &nums, int k);
double lazy_select(std::vector<double> &nums, int k);
bool check_lazy_function(double (*ptr_origin)(std::vector<double> &, int), double (*ptr_lazy)(std::vector<double> &, int), double (*ptr_quick)(std::vector<double> &, int k));
void run_experiments(std::vector<double (*)(std::vector<double> &, int)> functions);

// 第二个实验所需函数
template <typename T>
void origin_quick_sort(std::vector<T> &nums);
template <typename T>
void modify_quick_sort(std::vector<T> &nums);
void generateData(std::vector<std::vector<int>> &nums, int size);

#endif // !__LAB2_H__
