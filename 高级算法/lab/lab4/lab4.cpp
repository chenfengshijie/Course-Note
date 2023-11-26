#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include "lab4.h"

std::vector<Edge> generateRandomGraph(int n)
{
    std::vector<Edge> graph;
    std::srand(static_cast<unsigned int>(std::time(nullptr))); // 设置随机种子

    for (int u = 0; u < n; ++u)
    {
        for (int v = u + 1; v < n; ++v)
        {
            Edge edge;
            edge.u = u;
            edge.v = v;
            edge.weight = static_cast<double>(std::rand()) / RAND_MAX; // 生成随机权值在 (0,1) 范围内
            graph.push_back(edge);
        }
    }

    return graph;
}

void printGraph(const std::vector<Edge> &graph)
{
    for (const Edge &edge : graph)
    {
        std::cout << "Edge: " << edge.u << " - " << edge.v << ", Weight: " << edge.weight << std::endl;
    }
}
int find_father(int x, int *fa)
{
    if (fa[x] == x)
        return x;
    fa[x] = find_father(fa[x], fa);
    return fa[x];
}

double kruskal(std::vector<Edge> &graph, int n)
{
    sort(graph.begin(), graph.end(), [](Edge &a, Edge &b)
         { return a.weight < b.weight; });
    int cnt = 0;
    int *fa = new int[n + 10];
    for (int i = 0; i < n; i++)
        fa[i] = i;
    int p = 0;
    double ans = 0.0;
    while (cnt < n - 1)
    {
        auto tmp = graph[p];
        p++;
        int ls = tmp.u, rs = tmp.v;
        int fls = find_father(ls, fa);
        int frs = find_father(rs, fa);
        if (fls != frs)
        {
            ans += tmp.weight;
            fa[fls] = fa[frs];
            ++cnt;
        }
    }
    return ans;
}
