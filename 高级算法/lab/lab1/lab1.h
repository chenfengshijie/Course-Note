/**
 * @file lab1.h
 * @author Froze Chen (chenfengandchenyu@foxmail.com) 陈峰
 * @brief
 * @version 0.1
 * @date 2023-05-20
 *
 * @copyright Copyright (c) 2023
 *
 */

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
/**
 * @brief Reads data from a file and populates the documents vector.
 *
 * This function reads data from a file and populates the documents vector with unordered sets of integers.
 *
 * @param[in,out] documents The vector of unordered sets of integers to be populated.
 * @param[in] filename The name of the file to read data from.
 *
 * @return The number of sets
 */
int reader_faster(std::vector<std::unordered_set<int>> &documents, const char *filename);

/**
 * @brief Computes the naive similarity pairs based on a given threshold.
 *
 * This function computes the naive similarity pairs for a collection of documents based on a given threshold.
 *
 * @param[in] documents The vector of unordered sets of integers representing the documents.
 * @param[in] threshold The threshold value for similarity calculation (default is 0.5).
 *
 * @return A vector of pairs representing the similarity pairs.
 */
std::vector<std::pair<int, int>> naive_sim_threshold(std::vector<std::unordered_set<int>> &documents, double threshold = 0.5);

/**
 * @brief Generates random hash values.
 *
 * This function generates random hash values for the miniHash algorithm with Locality Sensitive Hashing (LSH).
 *
 * @param[in] a The 'a' parameter for hash generation.
 * @param[in] b The 'b' parameter for hash generation.
 * @param[in] p The 'p' parameter for hash generation.
 * @param[in] n The number of hash values to generate.
 * @param[in] data The vector of integers used for hash generation.
 *
 * @return A vector of random hash values.
 */
std::vector<int> generate_random_hash(int a, int b, int p, int n, const std::vector<int> &data);

/**
 * @brief Computes the minhash signature for a document.
 *
 * This function computes the minhash signature for a document based on a set of hash functions.
 *
 * @param[in] hash_functions The vector of hash functions used for minhash computation.
 * @param[in] document The unordered set of integers representing the document.
 *
 * @return A vector of minhash signature values.
 */
std::vector<int> compute_minhash_signature(const std::vector<std::vector<int>> &hash_functions, const std::unordered_set<int> &document);

/**
 * @brief Computes the Jaccard similarity between two minhash signatures.
 *
 * This function computes the Jaccard similarity between two minhash signatures.
 *
 * @param[in] sig1 The first minhash signature.
 * @param[in] sig2 The second minhash signature.
 *
 * @return The Jaccard similarity value.
 */
double compute_jaccard_similarity(const std::vector<int> &sig1, const std::vector<int> &sig2);

/**
 * @brief Performs Locality Sensitive Hashing (LSH) on a collection of minhash signatures.
 *
 * This function performs Locality Sensitive Hashing (LSH) on a collection of minhash signatures.
 *
 * @param[in] signatures The vector of minhash signatures.
 * @param[in] bands The number of bands for LSH.
 * @param[in] rows The number of rows per band for LSH.
 *
 * @return An unordered map with bucket keys and corresponding bucket values.
 */
std::unordered_map<std::string, std::vector<int>> lsh(const std::vector<std::vector<int>> &signatures, int bands, int rows);

/**
 * @brief Computes similarity pairs based on a given threshold using LSH buckets.
 *
 * This function computes similarity pairs based on a given threshold using LSH buckets.
 *
 * @param[in] buckets The unordered map of LSH buckets.
 * @param[in] signatures The vector of minhash signatures.
 * @param[in] threshold The threshold value for similarity calculation (default is 0.5).
 *
 * @return A vector of pairs representing the similarity pairs.
 */
std::vector<std::pair<int, int>> compute_sim_threshold(std::unordered_map<std::string, std::vector<int>> &buckets, std::vector<std::vector<int>> signatures, double threshold = 0.5);

/**
 * @brief Computes the true Jaccard similarity between two sets.
 *
 * This function computes the true Jaccard similarity between two unordered sets.
 *
 * @param[in] set1 The first unordered set.
 * @param[in] set2 The second unordered set.
 *
 * @return The true Jaccard similarity value.
 */
double true_jaccard_similarity(const std::unordered_set<int> &set1, const std::unordered_set<int> &set2);

/**
 * @brief Runs an experiment for similarity calculation.
 *
 * This function runs an experiment for similarity calculation based on a collection of documents.
 *
 * @param[in] documents The vector of unordered sets of integers representing the documents.
 * @param[in] num_hash_functions The number of hash functions to use for minhash computation.
 * @param[in] threshold The threshold value for similarity calculation.
 * @param[in] n The size of the hash values to generate.
 */
void run_experiment(const std::vector<std::unordered_set<int>> &documents, int num_hash_functions, double threshold, int n);

#endif // !__LAB1_H__