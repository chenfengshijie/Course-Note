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
void run_experiments(std::vector<double (*)(std::vector<double> &, int)> functions);

// 第二个实验所需函数
void origin_quick_sort(std::vector<double> &nums);
void modify_quick_sort(std::vector<double> &nums);

#endif // !__LAB2_H__
