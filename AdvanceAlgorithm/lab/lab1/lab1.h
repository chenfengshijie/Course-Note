#ifndef __LAB1_H__
#define __LAB1_H__
#include <vector>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <algorithm>
#include <random>
#include <string>
#include <ctime>
// below is dataloader
int reader(std::vector<std::unordered_set<int>> &documents, const char *filename);
// below is naive sim
std::vector<std::pair<int, int>> naive_sim_threshold(std::vector<std::unordered_set<int>> &doucuments, double threshold = 0.5);

// below is miniHash with LSH
std::vector<int> generate_random_hash(int a, int b, int p, int n, const std::vector<int> &data);
std::vector<int> compute_minhash_signature(const std::vector<std::vector<int>> &hash_functions, const std::unordered_set<int> &document);
double compute_jaccard_similarity(const std::vector<int> &sig1, const std::vector<int> &sig2);
std::unordered_map<std::string, std::vector<int>> lsh(const std::vector<std::vector<int>> &signatures, int bands, int rows);
std::vector<std::pair<int, int>> compute_sim_threshold(std::unordered_map<std::string, std::vector<int>> &buckets, std::vector<std::vector<int>>, double threshold = 0.5);

// below is experiments
double true_jaccard_similarity(const std::unordered_set<int> &set1, const std::unordered_set<int> &set2);
void run_experiment(const std::vector<std::unordered_set<int>> &documents, int num_hash_functions, double threshold, int n);

#endif // !__LAB1_H__