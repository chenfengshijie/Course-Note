#include "Lab2.h"
#include <iostream>
#include <cmath>
#include <random>
#include <ctime>
#include <algorithm>
#include <chrono>
#include <string>
#include <fstream>
using std::vector;
/// @brief generate uniform data
/// @param n  nums of data
/// @param minVal range
/// @param maxVal range
/// @return data
std::vector<double> generateUniformDistribution(int n, double minVal, double maxVal)
{
    std::vector<double> data;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(minVal, maxVal);

    for (int i = 0; i < n; i++)
    {
        double val = dis(gen);
        data.push_back(val);
    }

    return data;
}

// 生成服从正态分布的随机数据
std::vector<double> generateNormalDistribution(int n, double mean, double stddev)
{
    std::vector<double> data;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dis(mean, stddev);

    for (int i = 0; i < n; i++)
    {
        double val = dis(gen);
        data.push_back(val);
    }

    return data;
}
/**
 * @brief Zipf 分布服从p(x) = C / x^s,  x >= 1，它是一种大部分数据集中于队首的分布，这很大概率会卡掉lazy_select.生成服从Zipf分布的随机数据
 *
 * @param n 数据及大小
 * @param alpha 公式中的s
 * @param numItems x的范围为[1,numItems]
 * @return std::vector<double>
 */
std::vector<double> generateZipfDistribution(int n, double alpha, int numItems)
{
    std::vector<double> data;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    // 生成Zipf分布的概率分布
    std::vector<double> probabilities(numItems);
    double norm = 0.0;
    for (int i = 1; i <= numItems; i++)
    {
        norm += 1.0 / pow(i, alpha);
    }
    for (int i = 0; i < numItems; i++)
    {
        probabilities[i] = (1.0 / pow(i + 1, alpha)) / norm;
    }

    // 生成Zipf分布的随机数据
    for (int i = 0; i < n; i++)
    {
        double val = 0.0;
        double randNum = dis(gen);
        for (int j = 0; j < numItems; j++)
        {
            val += probabilities[j];
            if (randNum <= val)
            {
                data.push_back(j + 1); // 数据从1开始计数
                break;
            }
        }
    }

    return data;
}
// sort_select
double select_after_sort(std::vector<double> &nums, int k)
{
    std::sort(nums.begin(), nums.end());
    return nums[k];
}
int partition(double arr[], int low, int high)
{
    int pivotIndex = rand() % (high - low + 1) + low;
    double pivot = arr[pivotIndex]; // 选择数组的第一个元素作为 pivot
    int i = low + 1;
    int j = high;

    while (true)
    {
        // 找到第一个比 pivot 大的元素
        while (i <= j && arr[i] < pivot)
        {
            i++;
        }
        // 找到第一个比 pivot 小的元素
        while (i <= j && arr[j] > pivot)
        {
            j--;
        }
        if (i > j)
        {
            break; // 如果 i > j，说明已经遍历完成
        }
        // 交换 arr[i] 和 arr[j]
        std::swap(arr[i], arr[j]);
        i++;
        j--;
    }

    // 将 pivot 放到正确的位置上
    std::swap(arr[low], arr[j]);
    return j; // 返回 pivot 的下标
}
double kthSmallest_nonrecursion(double arr[], int n, int k)
{
    int low = 0;
    int high = n - 1;
    while (low <= high)
    {
        int pivotIndex = partition(arr, low, high); // 获取 pivot 的下标
        if (k == pivotIndex)
        {
            return arr[pivotIndex]; // 如果 k 等于 pivot 的下标，直接返回 pivot
        }
        else if (k < pivotIndex)
        {
            high = pivotIndex - 1; // 在左侧子数组中继续查找
        }
        else
        {
            low = pivotIndex + 1; // 在右侧子数组中继续查找
        }
    }
    // 如果 k 超出数组的范围，则返回一个错误值（这里可以根据实际需求进行调整）
    return -1;
}
/**
 * @brief return k smallest element of arr in [low,high]
 *
 * @param arr array
 * @param low low index
 * @param high high index
 * @param k kth element
 * @return double the specific element
 */
double kthSmallest(double arr[], int low, int high, int k)
{
    if (low == high)
    {
        return arr[low]; // 当数组只有一个元素时，直接返回该元素
    }
    int pivotIndex = partition(arr, low, high); // 获取 pivot 的下标
    if (k == pivotIndex)
    {
        return arr[pivotIndex]; // 如果 k 等于 pivot 的下标，直接返回 pivot
    }
    else if (k < pivotIndex)
    {
        return kthSmallest(arr, low, pivotIndex - 1, k); // 在左侧子数组中继续查找
    }
    else
    {
        return kthSmallest(arr, pivotIndex + 1, high, k); // 在右侧子数组中继续查找
    }
}
double linear_select(std::vector<double> &nums, int k)
{
    // return kthSmallest(nums.data(),0,nums.size()-1,k);
    return kthSmallest_nonrecursion(nums.data(), nums.size(), k);
}
double lazy_select_function(std::vector<double> &nums, int k, bool &success)
{
    int select_num = ceil(std::pow(nums.size(), 0.75));
    std::shuffle(nums.begin(), nums.end(), std::default_random_engine{});
    std::vector<double> lazy_set;
    lazy_set.resize(select_num);
    for (int i = 0; i < select_num; i++)
        lazy_set[i] = nums[i];
    int x = ceil((static_cast<double>(k) / nums.size()) * select_num);
    int l = std::max(0, x - int(sqrt(nums.size()))), h = std::min(select_num - 1, x + int(sqrt(nums.size())));
    std::sort(lazy_set.begin(), lazy_set.end());
    double L = lazy_set[l], H = lazy_set[h];
    int rank_l = 0, rank_h = 0;
    std::vector<double> P;
    for (int i = 0; i < nums.size(); i++)
    {
        if (nums[i] < L)
            ++rank_l;
        if (nums[i] < H)
            ++rank_h;
        if (nums[i] >= L && nums[i] <= H)
        {
            P.push_back(nums[i]);
        }
    }
    if (!(k >= rank_l && k <= rank_h && P.size() <= 4 * select_num + 1))
    {
        success = false;
        return -1.0;
    }
    success = true;
    std::sort(P.begin(), P.end());
    return P[k - rank_l];
}
// lazy_select
double lazy_select(std::vector<double> &nums, int k)
{
    bool success = false;
    int cnt = 0;
    double ans = 0;
    while (!success)
    {
        ans = lazy_select_function(nums, k, success);
        ++cnt;
        if (cnt > 1000)
            return -1;
    }
    // printf("%d\n", cnt);
    return ans;
}
/**
 * @brief
 *
 * @param ptr_origin funtion ptr to sort_select
 * @param ptr_lazy function ptr to lazy_select
 * @param ptr_quick ...to quick_select
 * @return true pass test
 * @return false fail test
 */
bool check_lazy_function(double (*ptr_origin)(vector<double> &, int), double (*ptr_lazy)(vector<double> &, int), double (*ptr_quick)(vector<double> &, int k))
{
    double ans1, ans2, ans3;
    // bool success = false;
    auto val_num = generateUniformDistribution(1000, 1.0, 5000.0);
    int k = rand() % val_num.size();
    ans1 = ptr_origin(val_num, k);
    ans2 = ptr_quick(val_num, k);
    ans3 = ptr_lazy(val_num, k);
    int isRight = true;
    if (ans2 != ans1)
    {
        printf("quick_select is wrong");
        isRight = false;
    }
    if (ans3 != ans1)
    {
        printf("lazy_select is wrong");
        isRight = false;
    }
    return isRight;
}

/**
 * @brief function,follow the order:sort_select,quick_select,lazy_select in functions;
 */
void run_experiments(std::vector<double (*)(std::vector<double> &, int)> functions)
{
    clock_t st[3], end[3];
    double tot_time;
    int k = 100;
    auto test_single_function = [](vector<double> &nums, int k, double (*func)(vector<double> &, int)) -> double
    { return func(nums, k); };
    auto duration = [](clock_t st, clock_t en) -> double
    { return static_cast<double>(en - st) / (CLOCKS_PER_SEC); };
    FILE *fout;
    fout = fopen("experiments.txt", "w");

    // uniform
    printf("---------Uniform time--------\n");
    fprintf(fout, "---------Uniform time--------\n");
    auto nums = generateUniformDistribution(500000, 1.0, 10000.0);
    for (int i = 0; i < 3; i++)
    {
        switch (i)
        {
        case 0:
            fprintf(fout, "-----sort-----\n");
            printf("-----sort-----\n");
            break;
        case 1:
            fprintf(fout, "-----quick-----\n");
            printf("-----quick-----\n");
            break;
        case 2:
            fprintf(fout, "-----lazy-----\n");
            printf("-----lazy-----\n");
            break;

        default:
            break;
        }
        for (int k = 1; k <= nums.size(); k += 50000)
        {
            fprintf(fout, "%d ", k);
            printf("%d ", k);
            st[i] = clock();
            test_single_function(nums, k, functions[i]);
            end[i] = clock();
            printf("%lf\n", duration(st[i], end[i]));
            fprintf(fout, "%lf\n", duration(st[i], end[i]));
        }
    }

    nums = generateNormalDistribution(500000, 1.0, 10000.0);
    fprintf(fout, "-------Normal time-------\n");
    printf("-------Normal time-------\n");
    for (int i = 0; i < 3; i++)
    {
        switch (i)
        {
        case 0:
            fprintf(fout, "-----sort-----\n");
            printf("-----sort-----\n");
            break;
        case 1:
            fprintf(fout, "-----quick-----\n");
            printf("-----quick-----\n");
            break;
        case 2:
            fprintf(fout, "-----lazy-----\n");
            printf("-----lazy-----\n");
            break;

        default:
            break;
        }
        for (int k = 1; k <= nums.size(); k += 50000)
        {
            fprintf(fout, "%d ", k);
            printf("%d ", k);
            st[i] = clock();
            test_single_function(nums, k, functions[i]);
            end[i] = clock();
            printf("%lf\n", duration(st[i], end[i]));
            fprintf(fout, "%lf\n", duration(st[i], end[i]));
        }
    }

    nums = generateZipfDistribution(500000, 1.0, 100000);
    fprintf(fout, "--------Zipf time--------\n");
    printf("--------Zipf time--------\n");
    for (int i = 0; i < 3; i++)
    {
        switch (i)
        {
        case 0:
            fprintf(fout, "-----sort-----\n");
            printf("-----sort-----\n");
            break;
        case 1:
            fprintf(fout, "-----quick-----\n");
            printf("-----quick-----\n");
            break;
        case 2:
            fprintf(fout, "-----lazy-----\n");
            printf("-----lazy-----\n");
            break;

        default:
            break;
        }
        for (int k = 1; k <= nums.size(); k += 50000)
        {
            fprintf(fout, "%d ", k);
            printf("%d ", k);
            st[i] = clock();
            test_single_function(nums, k, functions[i]);
            end[i] = clock();
            printf("%lf\n", duration(st[i], end[i]));
            fprintf(fout, "%lf\n", duration(st[i], end[i]));
        }
    }

    return;
}
//!! below is the second project

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
    double pivot = arr[high];
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
    int pivotValue = randomPivot(nums, left, right);

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

void modify_quick_sort(std::vector<double> &nums)
{
    modified_quicksort(nums, 0, nums.size() - 1);
    return;
}
/**
 * @brief generate 11 datasets,
 *
 * @param datasets return through reference
 * @param size the size of data
 */
void generateData(vector<vector<int>> &datasets, int size)
{
    using namespace std;
    datasets.resize(11);
    const int N = size;

    // 1. 无序数组，数组元素各不相同
    vector<int> nums1(N);
    for (int i = 0; i < N; i++)
    {
        nums1[i] = i + 1;
    }
    shuffle(nums1.begin(), nums1.end(), default_random_engine(chrono::system_clock::now().time_since_epoch().count()));
    datasets[0] = nums1;

    // 2. 其他元素各不相同，一个元素占整个数组的10%到100%
    for (int i = 1; i <= 10; i++)
    {
        int numCount = N * i / 10;
        vector<int> nums2(numCount);
        for (int j = 0; j < numCount - 1; j++)
        {
            nums2[j] = j + 1;
        }
        nums2[numCount - 1] = numCount * 10;
        shuffle(nums2.begin(), nums2.end(), default_random_engine(chrono::system_clock::now().time_since_epoch().count()));
        datasets[i] = nums2;
    }
}
