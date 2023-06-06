/**
 * @file lab3.h
 * @brief SkipList 数据结构的声明和实现
 * @author chen feng
 * @date 2023-04-29
 */

/**
 * @class Node
 * @brief 跳表节点类
 *
 * 跳表节点类包含了节点的键和指向其它节点的指针。
 */
#include <iostream>
#include <cstdlib>
#include <vector>
#include <bitset>
#include <functional>
using std::cout;
using std::endl;
using std::vector;

// * 下面是跳表的抽象数据结构的实现，为实验三的第一个项目

template <typename T>
class Node
{
public:
    T key;                  ///< 节点的键
    vector<Node *> forward; ///< 指向其它节点的指针数组

    /**
     * @brief 构造一个新的跳表节点
     * @param key 节点的键
     * @param level 节点的层数
     */
    Node(const T &key, unsigned int level) : key(key), forward(level + 1, nullptr){};
};

/**
 * @class SkipList
 * @brief 跳表数据结构类
 *
 * 跳表类提供了插入、删除、搜索和显示跳表的功能。
 */
template <typename T>
class SkipList
{
public:
    /**
     * @brief 构造一个新的跳表实例
     * @param max_level 跳表允许的最大层数
     * @param p 用于确定新节点层数的概率
     */
    SkipList(int max_level, float p);

    /**
     * @brief 销毁跳表实例
     */
    ~SkipList();

    /**
     * @brief 随机生成节点层数
     * @return 返回随机生成的节点层数
     */
    int random_level();

    /**
     * @brief 创建一个新的节点
     * @param key 节点的键
     * @param level 节点的层数
     * @return 返回新创建的节点指针
     */
    Node<T> *create_node(const T &key, int level);

    /**
     * @brief 向跳表中插入一个新的节点
     * @param key 要插入的节点的键
     */
    void insert(const T &key);

    /**
     * @brief 从跳表中删除一个节点
     * @param key 要删除的节点的键
     */
    void delete_key(const T &key);

    /**
     * @brief 在跳表中搜索一个节点
     * @param key 要搜索的节点的键
     * @return 返回搜索到的节点指针,or nullptr
     */
    Node<T> *search(const T &key);
    /**
     * @brief 范围查询[l,r]
     *
     * @param key1 l
     * @param key2 r
     * @return vector<vector<T*>> 指针
     */
    vector<Node<T> *> range_query(const T &key1, const T &key2);

    void range_delete(const T &key1, const T &key2);
    /**
     * @brief 显示跳表的内容
     */
    void display();

private:
    unsigned int max_level; ///< 跳表允许的最大层数
    float p;                ///< 用于确定新节点层数的概率
    unsigned int level;     ///< 当前跳表的层数
    Node<T> *header;        ///< 跳表的头节点
};

// 跳表构造函数
template <typename T>
SkipList<T>::SkipList(int max_level, float p) : max_level(max_level), p(p), level(0)
{
    header = new Node<T>(T(), max_level);
}

// 跳表析构函数
template <typename T>
SkipList<T>::~SkipList()
{
    delete header;
}

// 随机生成节点层数
template <typename T>
int SkipList<T>::random_level()
{
    float r = static_cast<float>(rand()) / RAND_MAX;
    int lvl = 0;
    while (r < p && lvl < max_level)
    {
        lvl++;
        r = static_cast<float>(rand()) / RAND_MAX;
    }
    return lvl;
}

// 创建一个节点
template <typename T>
Node<T> *SkipList<T>::create_node(const T &key, int level)
{
    return new Node<T>(key, level);
}

// 插入节点
template <typename T>
void SkipList<T>::insert(const T &key)
{
    vector<Node<T> *> update(max_level + 1);
    Node<T> *x = header;

    for (int i = level; i >= 0; i--)
    {
        while (x->forward[i] != nullptr && x->forward[i]->key < key)
        {
            x = x->forward[i];
        }
        update[i] = x;
    }

    int new_level = random_level();

    if (new_level > level)
    {
        for (int i = level + 1; i <= new_level; i++)
        {
            update[i] = header;
        }
        level = new_level;
    }

    x = create_node(key, new_level);

    for (int i = 0; i <= new_level; i++)
    {
        x->forward[i] = update[i]->forward[i];
        update[i]->forward[i] = x;
    }
}

template <typename T>
Node<T> *SkipList<T>::search(const T &key)
{
    Node<T> *x = header;
    for (int i = level; i >= 0; i--)
    {
        while (x->forward[i] != nullptr && x->forward[i]->key < key)
        {
            x = x->forward[i];
        }
    }

    x = x->forward[0];

    return x != nullptr && x->key == key ? x : nullptr;
}

template <typename T>
vector<Node<T> *> SkipList<T>::range_query(const T &left, const T &right)
{
    vector<Node<T> *> result;
    Node<T> *x = header;
    for (int i = level; i >= 0; --i)
    {
        while (x->forward[i] != nullptr && x->forward[i]->key < left)
        {
            x = x->forward[i];
        }
    }
    x = x->forward[0];
    while (x != nullptr && x->key <= right)
    {
        result.push_back(x);
        x = x->forward[0];
    }
    return result;
}
// 删除节点
template <typename T>
void SkipList<T>::delete_key(const T &key)
{
    vector<Node<T> *> update(max_level + 1);
    Node<T> *x = header;

    for (int i = level; i >= 0; i--)
    {
        while (x->forward[i] != nullptr && x->forward[i]->key < key)
        {
            x = x->forward[i];
        }
        update[i] = x;
    }

    x = x->forward[0];
    if (x->key == key)
    {
        for (int i = 0; i <= level; i++)
        {
            if (update[i]->forward[i] != x)
            {
                break;
            }
            update[i]->forward[i] = x->forward[i];
        }

        delete x;

        while (level > 0 && header->forward[level] == nullptr)
        {
            level--;
        }
    }
}
// TODO:由于是范围删除，所以可以进行一定的优化，例如删除完所有元素之后才进行层数的更新
template <typename T>
void SkipList<T>::range_delete(const T &left, const T &right)
{
    vector<Node<T> *> nodes_to_delete = range_query(left, right);
    for (auto node : nodes_to_delete)
    {
        delete_key(node->key);
    }
    // Node<T> *x = header;
    // vector<std::pair<Node<T> *,Node<T>*>> update(max_level + 1);
    // for (int i = level; i >= 0; --i)
    //{
    //     while (x->forward[i] != nullptr && x->forward[i]->key < left)
    //     {
    //         x = x->forward[i];
    //     }
    //     update[i].first = x;
    //     while (x->forward[i] != nullptr && x->forward[i]->key < right)
    //     {
    //         x = x->forward[i];
    //     }
    //     update[i].second = x;
    // }
    // x = x->forward[0];
    // for (int i = 0; i < level; i++)
    //{
    //     if (update[i].first->forward[i]
    //         update[i].first->forward[i] = update[i].second->forward[i];
    //     else
    //         update[i].first->forward[i] = nullptr;
    // }
    // auto nex = x->forward[0];
    // while (x != nullptr && x->key <= right)
    //{
    //     delete x;
    //     x = nex;
    //     nex = nex->forward[0];
    // }
    // while (level > 0 && header->forward[level] == nullptr)
    //{
    //     level--;
    // }
}

template <typename T>
void SkipList<T>::display()
{
    for (int i = 0; i <= level; i++)
    {
        Node<T> *x = header->forward[i];
        cout << "Level " << i << ": ";
        while (x != nullptr)
        {
            cout << x->key << " ";
            x = x->forward[i];
        }
        cout << endl;
    }
}

void run_experiment1();

// * 下面是实验三第二个项目Bloom Filter的实现

const int BITSET_SIZE = 10000; // 位数组长度
const int HASH_NUM = 5;        // 哈希函数个数
template <typename T>
class BloomFilter
{
public:
    BloomFilter()
    {
        bitset_.reset(); // 初始化位数组，全部置为0
    }
    /**
     * @brief add elements
     *
     * @param key the key need to be added
     */
    template <typename T>
    void add(T key)
    {
        for (int i = 0; i < HASH_NUM; i++)
        {
            uint32_t hash_value = std::hash<T>{}(key + i); // 使用 std::hash 生成哈希值
            uint32_t index = hash_value % BITSET_SIZE;
            bitset_.set(index); // 将对应的位数组位置置为1
        }
    }
    /**
     * @brief whether contain a specific key
     *
     * @param key a key
     * @return true Found
     * @return false Not found
     */
    template <typename T>
    bool contains(T key) const
    {
        for (int i = 0; i < HASH_NUM; i++)
        {
            uint32_t hash_value = std::hash<T>{}(key + i); // 使用 std::hash 生成哈希值
            uint32_t index = hash_value % BITSET_SIZE;
            if (!bitset_.test(index))
            { // 如果有任意一个位置为0，则表示元素不存在于集合中
                return false;
            }
        }
        return true; // 所有位置都为1，则表示元素可能存在于集合中
    }

private:
    // the bit array
    std::bitset<BITSET_SIZE> bitset_;
};

//
