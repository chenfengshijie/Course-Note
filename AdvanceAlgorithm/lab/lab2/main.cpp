#include "lab2.h"
#include <chrono>

int main()
{
    // 第一个实验
    std::vector<double (*)(std::vector<double> &, int)> functions;
    functions.push_back(select_after_sort);
    functions.push_back(linear_select);
    functions.push_back(lazy_select);

    // bool test = check_lazy_function(select_after_sort, lazy_select, linear_select);
    run_experiments(functions);
    // 第二个实验

    return 0;
}