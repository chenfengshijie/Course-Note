#include <iostream>
#include "lab4.h"

int main(int, char **)
{
    int begin_n = 16;
    int times = 1000;
    for (int i = begin_n; i <= 2048; i <<= 1, times /= 2)
    {
        double val = 0.0;
        for (int j = 0; j <= times; j++)
        {
            auto graph = generateRandomGraph(i);
            val += kruskal(graph, i);
        }
        printf("%d: %.4lf\n", i, val / times);
    }
    auto g1 = generateRandomGraph(10);
    printGraph(g1);
}
