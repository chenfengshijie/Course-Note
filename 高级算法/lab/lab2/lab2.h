#ifndef __LAB2_H__
#define __LAB2_H__
#include <vector>
#include <string>
#include <unordered_map>
#include <iostream>
// 第一个实验所需函数
std::vector<double> generateUniformDistribution(int n, double minVal, double maxVal);
std::vector<double> generateNormalDistribution(int n, double mean, double stddev);
std::vector<double> generateZipfDistribution(int n, double alpha, int numItems);
double select_after_sort(std::vector<double> &nums, int k);
double linear_select(std::vector<double> &nums, int k);
double lazy_select(std::vector<double> &nums, int k);
bool check_lazy_function(double (*ptr_origin)(std::vector<double> &, int), double (*ptr_lazy)(std::vector<double> &, int), double (*ptr_quick)(std::vector<double> &, int k));
void run_experiments(std::vector<double (*)(std::vector<double> &, int)> functions);

// 第二个实验函数

template <typename T>
void origin_quick_sort(std::vector<T> &nums);
template <typename T>
void modify_quick_sort(std::vector<T> &nums);
void generateData(std::vector<std::vector<int>> &nums, int size);
void run_experiment_quicksort(std::vector<void (*)(std::vector<int> &)>);

// ! 下面是模板函数，所以必须实现在头文件
using std::vector; // 在工程上不应该这样写，不过实验就算了
// extern int stack_depth; // 用于检测栈深度，防止栈溢出
//! 上述检测栈深度防止溢出会误伤正常的函数运行，所以下述我通过检测运行时间来检测栈。
extern clock_t st, en;
/**
 * @brief Function to partition the array and return the partition index.
 * @param arr The array to be sorted.
 * @param low The starting index of the subarray to be sorted.
 * @param high The ending index of the subarray to be sorted.
 * @return The partition index.
 */
template <typename T>
int partitionForSecond(vector<T> &arr, int low, int high)
{
    T pivot = randomPivot(arr, low, high);
    int i = low - 1;

    for (int j = low; j < high; j++)
    {
        if (arr[j] <= pivot)
        {
            i++;
            std::swap(arr[i], arr[j]);
        }
    }

    std::swap(arr[i + 1], arr[high]);
    return i + 1;
}

/**
 * @brief Function to perform quicksort recursively on the subarrays.
 * @param arr The array to be sorted.
 * @param low The starting index of the subarray to be sorted.
 * @param high The ending index of the subarray to be sorted.
 */
template <typename T>
void quicksort(vector<T> &arr, int low, int high)
{
    if (low < high)
    {
        int p = partitionForSecond(arr, low, high);
        // stack_depth++;
        //  if (stack_depth >= 5000)
        //      throw std::runtime_error("function call stack overflow");
        clock_t check = clock();
        if (1.00 * (check - st) / CLOCKS_PER_SEC > 10.0)
            throw std::runtime_error("function call stack overflow");
        quicksort(arr, low, p - 1);
        quicksort(arr, p + 1, high);
    }
}
/**
 * @brief origin quich sort
 *
 * @param nums given array
 */
template <typename T>
void origin_quick_sort(std::vector<T> &nums)
{
    quicksort(nums, 0, nums.size() - 1);
}

// * below is modified_quick_sort,3 ways to improve quick_sort
// * random select pivot
// * using insert_sort on array consists of less 5 elements
// * 3 ways quick sort instead of 2 ways

// 随机选择pivot
template <typename T>
T randomPivot(vector<T> &nums, int left, int right)
{
    int pivotIndex = rand() % (right - left + 1) + left;
    T pivotValue = nums[pivotIndex];
    std::swap(nums[pivotIndex], nums[right]);
    return pivotValue;
}

// 三路快排
template <typename T>
void modified_quicksort(vector<T> &nums, int left, int right)
{
    if (left >= right)
    {
        return;
    }
    if (right - left + 1 <= 5)
    { // 当子数组长度小于等于5时使用插入排序
        for (int i = left + 1; i <= right; i++)
        {
            T temp = nums[i];
            int j = i - 1;
            while (j >= left && nums[j] > temp)
            {
                nums[j + 1] = nums[j];
                j--;
            }
            nums[j + 1] = temp;
        }
        return;
    }

    // 随机选择pivot
    T pivotValue = randomPivot(nums, left, right);

    int lt = left, gt = right, i = left;
    while (i <= gt)
    {
        if (nums[i] < pivotValue)
        {
            std::swap(nums[i++], nums[lt++]);
        }
        else if (nums[i] > pivotValue)
        {
            std::swap(nums[i], nums[gt--]);
        }
        else
        {
            i++;
        }
    }

    modified_quicksort(nums, left, lt - 1);
    modified_quicksort(nums, gt + 1, right);
}
template <typename T>
void modify_quick_sort(std::vector<T> &nums)
{
    modified_quicksort(nums, 0, nums.size() - 1);
    return;
}

#endif // !__LAB2_H__
