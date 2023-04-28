#include "lab2.h"
#include <chrono>
#include <algorithm>

int main()
{
    // 第一个实验
    std::vector<double (*)(std::vector<double> &, int)> functions;
    functions.push_back(select_after_sort);
    functions.push_back(linear_select);
    functions.push_back(lazy_select);

    // bool test = check_lazy_function(select_after_sort, lazy_select, linear_select);
    // * run_experiments(functions);
    // 第二个实验
    // 对模板进行特例化
    std::vector<void (*)(std::vector<int> &)> functions_quicksort;
    auto ptr_origin = [](std::vector<int> &nums)
    { origin_quick_sort(nums); };
    functions_quicksort.push_back(ptr_origin);
    auto ptr_modified = [](std::vector<int> &nums)
    { modify_quick_sort(nums); };
    functions_quicksort.push_back(modify_quick_sort);
    auto ptr_stdsort = [](std::vector<int> &nums)
    { std::sort(nums.begin(), nums.end()); };
    // 开始实验
    functions_quicksort.push_back(ptr_stdsort);
    run_experiment_quicksort(functions_quicksort);
    return 0;
}