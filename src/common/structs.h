#pragma once

#include <string>
#include <vector>

using namespace std;

enum class RunType
{
    CPU,
    CL_CPU,
    CL_GPU
};

struct Run
{
    const string taskDescription;
    size_t size;

    Run(string taskDescription, size_t size)
        : taskDescription(taskDescription), size(size)
    {
    }
};

struct CPUIteration
{
    double runTime;
};

struct CPURun : public Run
{
    vector<CPUIteration> iterations;
    double runTimeMean;
    double runTimeDeviation;
    bool verificationResult;
    bool exceptionOccured;
    string exceptionMsg;

    CPURun(string taskDescription, size_t size)
        : Run(taskDescription, size), exceptionOccured(false)
    {
    }
};

struct CLIteration
{
    double uploadTime;
    double runTime;
    double downloadTime;
};

struct CLRunWithWGSize
{
    size_t wgSize;
    vector<CLIteration> iterations;
    double uploadTimeMean;
    double uploadTimeDeviation;
    double runTimeMean;
    double runTimeDeviation;
    double downloadTimeMean;
    double downloadTimeDeviation;
    bool verificationResult;
    bool exceptionOccured;
    string exceptionMsg;

    CLRunWithWGSize()
        : exceptionOccured(false)
    {
    }
};

struct CLRun : public Run
{
    //double initTime;
    //double cleanupTime;
    vector<CLRunWithWGSize> runsWithWGSize;
    vector<CLRunWithWGSize>::iterator fastest;
    double avgUploadTime;
    double avgRunTime;
    double avgDownloadTime;

    CLRun(string taskDescription, size_t size)
        : Run(taskDescription, size)
    {
    }
};