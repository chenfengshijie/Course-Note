#ifndef __LAB1_H__
#define __LAB1_H__
#include <vector>
bool read_file(std::vector<std::vector<int>> &data, const char *file_name);
vector<std::pair<int, int>> &naive(std::vector<std::vector<int>> &data, int c);
vector<std::pair<int, int>> &mini_hash(std::vector<std::vector<int>> &data, int c);
#endif