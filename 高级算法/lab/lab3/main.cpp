/**
 * @file main.cpp
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
using std::cout;
using std::endl;
int main()
{
    // srand(static_cast<unsigned>(time(0)));
    // SkipList<int> skip_list(4, 0.5);

    // skip_list.insert(3);
    // skip_list.insert(6);
    // skip_list.insert(7);
    // skip_list.insert(9);
    // skip_list.insert(12);
    // skip_list.insert(19);
    // skip_list.insert(17);

    // skip_list.display();

    // cout << "Search 7: " << (skip_list.search(7) ? "Found" : "Not found") << endl;
    // cout << "Search 4: " << (skip_list.search(4) ? "Found" : "Not found") << endl;

    // skip_list.delete_key(7);
    // if (skip_list.search(7))
    //     printf("false");
    // cout << "After deleting 7" << endl;
    // skip_list.display();
    BloomFilter<int> bf;
    bf.add(12);

    return 0;
}
