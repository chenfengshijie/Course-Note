#include "lab5.h"

#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <filesystem>

int main()
{
    namespace ch = std::chrono;
    std::string data_dir = R"(C:\Users\FrozeWorld\source\repos\chenfengshijie\AdvanceAlgorithm\Lab5\data)";
    std::vector<std::string> files;
    // get data files
    for (auto &entry : std::filesystem::directory_iterator(data_dir))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".txt")
        {
            std::cout << entry.path().string() << std::endl;
            files.push_back(entry.path().string());
        }
    }
    std::string method = "olken_weight";
    auto consume_time = [](ch::time_point<ch::steady_clock> st, ch::time_point<ch::steady_clock> en) -> long long
    { return ch::duration_cast<ch::microseconds>(en - st).count(); };
    // olken weight
    auto st = ch::high_resolution_clock::now();
    DataBase db_olken(4, files, method);
    auto en = ch::high_resolution_clock::now();
    std::vector<long long> init_time(3, 0);
    init_time[0] = consume_time(st, en);
    /// exact weight
    st = ch::high_resolution_clock::now();
    method = "exact_weight";
    DataBase db_exact(4, files, method);
    en = ch::high_resolution_clock::now();
    init_time[1] = consume_time(st, en);
    // random walk
    st = ch::high_resolution_clock::now();
    method = "random_walk";
    RandomWalkDataBase db_random(4, files, method);
    en = ch::high_resolution_clock::now();
    init_time[2] = consume_time(st, en);

    // calculate the sampling time
    std::vector<long long> sampling_time(3, 0);

    // olken weight
    st = ch::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++)
        while (db_olken.sample().size() == 0)
            ;
    en = ch::high_resolution_clock::now();
    sampling_time[0] = consume_time(st, en);
    // exact weight
    st = ch::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++)
        db_exact.sample();
    en = ch::high_resolution_clock::now();
    sampling_time[1] = consume_time(st, en);
    // random walk
    st = ch::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++)
        while (db_random.sample().size() == 0)
            ;
    en = ch::high_resolution_clock::now();
    sampling_time[2] = consume_time(st, en);

    // output the result
    std::ofstream out("result.txt");
    out << "init time" << std::endl;
    out << "olken weight: " << init_time[0] << std::endl;
    out << "exact weight: " << init_time[1] << std::endl;
    out << "random walk: " << init_time[2] << std::endl;
    out << "sampling time" << std::endl;
    out << "olken weight: " << sampling_time[0] << std::endl;
    out << "exact weight: " << sampling_time[1] << std::endl;
    out << "random walk: " << sampling_time[2] << std::endl;
    out.close();
    return 0;
}
