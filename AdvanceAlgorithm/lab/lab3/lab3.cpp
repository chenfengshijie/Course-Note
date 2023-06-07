/**
 * @file lab3.cpp
 * @author Froze Chen (chenfengandchenyu@foxmail.com)
 * @brief
 * @version 0.1
 * @date 2023-04-29
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "lab3.h"
#include <iostream>
#include <chrono>
#include <set>

void run_experiment1()
{
    using std::cout;
    using std::endl;
    using std::set;
    srand(static_cast<unsigned>(time(0)));
    SkipList<int> skip_list(4, 0.5);
    cout << "insert:" << endl;
    set<int> set1;

    // test insert single key
    auto start1 = std::chrono::high_resolution_clock::now();
    constexpr int test_num = 1000;
    for (int i = 0; i < test_num; i++)
    {
        skip_list.insert(i);
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
    cout << duration1.count() << ",";
    auto start2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < test_num; i++)
    {
        set1.insert(i);
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);
    cout << duration2.count() << endl;

    // test search single key
    cout << "search single:" << endl;
    start1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < test_num; i++)
    {
        skip_list.search(i);
    }
    end1 = std::chrono::high_resolution_clock::now();
    duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
    cout << duration1.count() << ",";

    start2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < test_num; i++)
    {
        set1.find(i);
    }
    end2 = std::chrono::high_resolution_clock::now();
    duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);
    cout << duration2.count() << endl;
    // test range search
    cout << "range search:" << endl;
    start1 = std::chrono::high_resolution_clock::now();
    int left, right;
    for (int i = 0; i < test_num; i++)
    {
        left = std::rand() % test_num;
        right = std::rand() % test_num;
        if (left > right)
            std::swap(left, right);

        skip_list.range_query(left, right);
    }
    end1 = std::chrono::high_resolution_clock::now();
    duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
    cout << duration1.count() << ",";

    start2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < test_num; i++)
    {
        left = std::rand() % test_num;
        right = std::rand() % test_num;
        if (left > right)
            std::swap(left, right);
        auto s1 = set1.lower_bound(left);
        auto s2 = set1.upper_bound(right);
        while (s1 != s2 && s1 != set1.end())
        {
            ++s1;
        }
    }
    end2 = std::chrono::high_resolution_clock::now();
    duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);
    cout << duration2.count() << endl;
    // test delete single key
    cout << "delete single";

    start1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < test_num; i++)
    {
        skip_list.delete_key(i);
    }
    end1 = std::chrono::high_resolution_clock::now();
    duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
    cout << duration1.count() << ",";

    start2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < test_num; i++)
    {
        set1.erase(i);
    }
    end2 = std::chrono::high_resolution_clock::now();
    duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);
    cout << duration2.count() << endl;

    ///<---------------------------->

    // test delete range
    cout << "delete range" << endl;
    for (int i = 0; i < test_num; i++)
    {
        skip_list.insert(i);
        set1.insert(i);
    }
    auto test_times = 10000;
    start1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < test_times; i += 10)
    {
        skip_list.range_delete(i, i + 10);
    }
    end1 = std::chrono::high_resolution_clock::now();
    duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
    cout << duration1.count() << ",";

    start2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < test_times; i += 10)
    {
        auto s1 = set1.lower_bound(i);
        auto s2 = set1.upper_bound(i + 10);
        set1.erase(s1, s2);
    }
    end2 = std::chrono::high_resolution_clock::now();
    duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);
    cout << duration2.count() << endl;
    return;
}

void run_experiment2()
{

    using std::cout;
    using std::endl;
    BloomFilter<int> bloom_filter;
    constexpr int test_num = 100000;
    cout << "insert:";
    auto start1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < test_num; i++)
    {
        bloom_filter.add(i);
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
    cout << duration1.count() << endl;
}