#pragma once

#include <vector>
#include <string>

#include "structs.h"

using namespace std;

class ConsoleWriter final
{
public:
    static const int FLOAT_PRECISION;

    void beginOutput(size_t iterations, vector<size_t> sizes, string typeName);
    void endOutput(double seconds);

    void beginAlgorithm(string algorithmName, RunType runType, double initTime = -1.0);
    void endAlgorithm(double cleanupTime = -1.0);

    void writeRun(const CPURun& run);
    void writeRun(const CLRun& run);
};

