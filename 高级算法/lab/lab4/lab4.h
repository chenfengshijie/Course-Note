#ifndef __LAB4_H__
#define __LAB4_H__
#include <vector>

struct Edge
{
    int u, v;
    double weight;
};
/**
 * @brief generate a n full graph
 *
 * @param n the num of point
 * @return std::vector<Edge> the graph
 */
std::vector<Edge> generateRandomGraph(int n);
/**
 * @brief kruskal to compute min-generate tree
 *
 * @param graph the graph
 * @param n
 * @return double the tot val of mgt
 */
double kruskal(std::vector<Edge> &graph, int n);
/**
 * @brief print the info of a graph
 *
 * @param graph
 */
void printGraph(const std::vector<Edge> &graph);
#endif