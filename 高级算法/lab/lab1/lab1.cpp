#include "lab1.h"
#include <vector>
#include <fstream>

using std::vector;
bool read_file(vector<vector<int>> &data, const char *file_name)
{
    data.clear();
    std::ifstream fin(file_name);
    int a, b;
    vector<int> tmp;
    int cnt = 0;
    tmp.push_back(0);
    while (!fin.eof())
    {
        fin >> a >> b;
        if (a != cnt)
        {
            cnt = a;
            tmp[0] = tmp.size() - 1;
            data.push_back(tmp);
            tmp.clear();
            tmp.push_back(0);
            tmp.push_back(b);
        }
    }
    fin.close();
    return true;
}
vector<std::pair<int, int>> &naive(vector<vector<int>> &data, int c)
{
    vector<std::pair<int, int>> *ans = new vector<std::pair<int, int>>;
    int n = data.size();
    int cup = 0, cap = 0;
    for (int i = 1; i < n; i++)
    {
        cup = 0, cap = 0;
        for (int j = i + 1; j < n; j++)
        {
        }
    }
}