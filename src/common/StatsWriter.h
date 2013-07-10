#pragma once

#include <string>
#include <vector>
#include <fstream>

#include "structs.h"

using namespace std;

class StatsWriter final
{
public:
    void beginFile(string fileName, char separator = ';');
    void endFile(double seconds);

    void beginAlgorithm(string algorithmName, RunType runType, double initTime = -1.0);
    void endAlgorithm(double cleanupTime = -1.0);

    void writeRun(const CPURun& run);
    void writeRun(const CLRun& run);

private:
    ofstream file;
    char sep;
};

