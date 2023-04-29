#include "lab3.h"

// * 下面是实验三第二个项目Bloom Filter的实现

const int BITSET_SIZE = 10000; // 位数组长度
const int HASH_NUM = 5;        // 哈希函数个数

class BloomFilter
{
public:
    BloomFilter()
    {
        bitset_.reset(); // 初始化位数组，全部置为0
    }

    void add(int key)
    {
        for (int i = 0; i < HASH_NUM; i++)
        {
            uint32_t hash_value = std::hash<int>{}(key + i); // 使用 std::hash 生成哈希值
            uint32_t index = hash_value % BITSET_SIZE;
            bitset_.set(index); // 将对应的位数组位置置为1
        }
    }

    bool contains(int key) const
    {
        for (int i = 0; i < HASH_NUM; i++)
        {
            uint32_t hash_value = std::hash<int>{}(key + i); // 使用 std::hash 生成哈希值
            uint32_t index = hash_value % BITSET_SIZE;
            if (!bitset_.test(index))
            { // 如果有任意一个位置为0，则表示元素不存在于集合中
                return false;
            }
        }
        return true; // 所有位置都为1，则表示元素可能存在于集合中
    }

private:
    std::bitset<BITSET_SIZE> bitset_;
};