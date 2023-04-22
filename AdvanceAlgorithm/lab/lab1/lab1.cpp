#include "lab1.h"
#include <chrono>
#include <iostream>
#include <filesystem>
int reader(std::vector<std::unordered_set<int>> &documents, const char *filename)
{
    int num_max = -1;
    std::ifstream fin(filename);
    if (!fin.is_open())
    {
        std::cerr << "can't open file" << filename;
        return false;
    }

    // 预分配一个足够大的向量
    const int num_documents = 100000; // 根据数据集的大小适当调整
    documents.reserve(num_documents);

    int a = 0, b, tmp = 0;
    std::unordered_set<int> document;
    while (fin >> a >> b)
    {
        num_max = std::max(num_max, b);
        if (a != tmp)
        {
            documents.emplace_back(std::move(document));
            document.clear();
            document.insert(b);
            tmp = a;
        }
        else
        {
            document.insert(b);
        }
    }

    fin.close();
    return num_max;
}

// below is naive sim
std::vector<std::pair<int, int>> naive_sim_threshold(std::vector<std::unordered_set<int>> &doucuments, double threshold)
{
    int n = doucuments.size();
    int count = 0;
    std::vector<std::pair<int, int>> ans;
    for (int iter1 = 0; iter1 < n; iter1++)
        for (int iter2 = iter1 + 1; iter2 < n; iter2++)
        {
            count = 0;
            for (auto &ix : doucuments[iter1])
                for (auto &iy : doucuments[iter2])
                    if (ix == iy)
                        ++count;
            if (static_cast<double>(count) / (doucuments[iter1].size() + doucuments[iter2].size() - count) >= threshold)
                ans.push_back(std::make_pair(iter1, iter2));
        }
    return ans;
}

// below is minihash
// 随机哈希函数
std::vector<int> generate_random_hash(int a, int b, int p, int n, const std::vector<int> &data)
{
    std::vector<int> hashed_values;
    for (int item : data)
    {
        int hash_value = (a * item + b) % p;
        hashed_values.push_back(hash_value % n);
    }
    return hashed_values;
}

// 计算minhash签名
std::vector<int> compute_minhash_signature(const std::vector<std::vector<int>> &hash_functions, const std::unordered_set<int> &document)
{
    std::vector<int> signature;
    for (const auto &hash_function : hash_functions)
    {
        int min_hash = INT_MAX;
        for (int item : document)
        {
            min_hash = std::min(min_hash, hash_function[item]);
        }
        signature.push_back(min_hash);
    }
    return signature;
}

// 计算Jaccard相似度
double compute_jaccard_similarity(const std::vector<int> &sig1, const std::vector<int> &sig2)
{
    int count = 0;
    for (size_t i = 0; i < sig1.size(); ++i)
    {
        if (sig1[i] == sig2[i])
        {
            count++;
        }
    }
    return static_cast<double>(count) / sig1.size();
}

// LSH
std::unordered_map<std::string, std::vector<int>> lsh(const std::vector<std::vector<int>> &signatures, int bands, int rows)
{
    std::unordered_map<std::string, std::vector<int>> buckets;
    for (size_t i = 0; i < signatures.size(); ++i)
    {
        for (int band = 0; band < bands; ++band)
        {
            std::string bucket_key = "";
            for (int row = 0; row < rows; ++row)
            {
                bucket_key += std::to_string(signatures[i][band * rows + row]) + std::to_string(band);
            }
            buckets[bucket_key].push_back(i);
        }
    }
    return buckets;
}

std::vector<std::pair<int, int>> compute_sim_threshold(std::unordered_map<std::string, std::vector<int>> &buckets, std::vector<std::vector<int>> signatures, double threshold)
{
    std::set<std::pair<int, int>> checked_pairs;
    std::vector<std::pair<int, int>> ans;
    for (const auto &bucket : buckets)
    {
        const auto &doc_ids = bucket.second;
        if (doc_ids.size() > 1)
        {
            for (size_t i = 0; i < doc_ids.size(); ++i)
            {
                for (size_t j = i + 1; j < doc_ids.size(); ++j)
                {
                    int doc_id1 = doc_ids[i];
                    int doc_id2 = doc_ids[j];

                    // 如果已经检查过该对文档，跳过
                    if (checked_pairs.find({doc_id1, doc_id2}) != checked_pairs.end() || doc_id1 == doc_id2)
                    {
                        continue;
                    }
                    checked_pairs.insert({doc_id1, doc_id2});

                    double similarity = compute_jaccard_similarity(signatures[doc_id1], signatures[doc_id2]);
                    if (similarity >= threshold)
                    {
                        std::cout << "文档 " << doc_id1 << " 和文档 " << doc_id2 << " 的相似度为: " << similarity << std::endl;
                        ans.push_back(std::make_pair(i, j));
                    }
                }
            }
        }
    }
    return ans;
}

// below is experiments

// 计算真实的Jaccard相似度
double true_jaccard_similarity(const std::unordered_set<int> &set1, const std::unordered_set<int> &set2)
{
    int count = 0;
    for (auto &v : set1)
    {
        if (set2.find(v) != set2.end())
            ++count;
    }
    return static_cast<double>(count) / (set1.size() + set2.size() - count);
}
/*
 * n - 元素个数
 *
 */
void run_experiment(const std::vector<std::unordered_set<int>> &documents, int num_hash_functions, double threshold, int n)
{

    std::chrono::seconds tot_time = static_cast<std::chrono::seconds>(0);
    int p = 0;
    // 获取大于n的素数p，构建hash函数
    for (int i = n + 1;; i++)
    {
        int k = ceil(sqrt(i));
        p = 0;
        for (int j = 2; j <= k; j++)
            if (i % j == 0)
            {
                p = -1;
                break;
            }
        if (p != -1)
        {
            p = i;
            break;
        }
    }
    int bands = 10;
    int rows = num_hash_functions / bands;
    auto st = std::chrono::high_resolution_clock::now();
    // 创建随机哈希函数集
    std::mt19937 gen(static_cast<unsigned>(time(nullptr)));
    std::uniform_int_distribution<> dis(1, p - 1);
    std::vector<std::vector<int>> hash_functions;
    std::vector<int> cnt;
    for (int i = 1; i <= n; i++)
        cnt.push_back(i);
    for (int i = 0; i < num_hash_functions; ++i)
    {
        int a = dis(gen);
        int b = dis(gen);
        hash_functions.push_back(generate_random_hash(a, b, p, n, cnt));
    }
    std::vector<std::vector<int>> signatures;
    for (const auto &document : documents)
    {
        signatures.push_back(compute_minhash_signature(hash_functions, document));
    }

    // 使用LSH构建桶
    std::unordered_map<std::string, std::vector<int>> buckets = lsh(signatures, bands, rows);
    int true_positive = 0;
    int false_positive = 0;
    int false_negative = 0;
    auto finish = std::chrono::high_resolution_clock::now();
    tot_time += std::chrono::duration_cast<std::chrono::seconds>(finish - st);
    for (const auto &bucket : buckets)
    {
        const auto &doc_ids = bucket.second;
        if (doc_ids.size() > 1)
        {
            for (size_t i = 0; i < doc_ids.size(); ++i)
            {
                for (size_t j = i + 1; j < doc_ids.size(); ++j)
                {
                    int doc_id1 = doc_ids[i];
                    int doc_id2 = doc_ids[j];

                    if (doc_id1 == doc_id2 || documents[doc_id1].size() * documents[doc_id2].size() == 0)
                    {
                        continue;
                    }
                    st = std::chrono::high_resolution_clock::now();
                    double minhash_similarity = compute_jaccard_similarity(signatures[doc_id1], signatures[doc_id2]);
                    finish = std::chrono::high_resolution_clock::now();
                    tot_time += std::chrono::duration_cast<std::chrono::seconds>(finish - st);

                    double true_similarity = true_jaccard_similarity(documents[doc_id1], documents[doc_id2]);

                    if (minhash_similarity >= threshold && true_similarity >= threshold)
                    {
                        true_positive++;
                    }
                    else if (minhash_similarity >= threshold && true_similarity < threshold)
                    {
                        false_positive++;
                    }
                    else if (minhash_similarity < threshold && true_similarity >= threshold)
                    {
                        false_negative++;
                    }
                }
            }
        }
    }

    double precision = static_cast<double>(true_positive) / (true_positive + false_positive);
    double recall = static_cast<double>(true_positive) / (true_positive + false_negative);
    std::ofstream fout("experiment.txt", std::ios::app);
    // fout<<true_positive<<"-"<<false_positive<<"-"<<false_negative<<std::endl;
    fout << "\n num_of_hashfunc:" << num_hash_functions << "\n"
         << "precision:" << precision << "\n"
         << "recall:" << recall
         << "time:" << tot_time.count()
         << std::endl;
    fout.close();
}
