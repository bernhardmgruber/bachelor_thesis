#include <iostream>

#include "Timer.h"

using namespace std;

Timer::Timer()
{
#ifdef _WIN32
    if (!QueryPerformanceFrequency(&frequency))
    {
        cerr << "High performance counter is not supported!" << endl;
        exit(0);
    }
#else
    /*if(clock_getres(CLOCK_MONOTONIC, &frequency) != -1)
    {
        cerr << "High resolution clock is not supported!" << endl;
        exit(0);
    }*/
#endif

    overhead = 0;
    start();
    overhead = stop();
}

void Timer::start()
{
#ifdef _WIN32
    QueryPerformanceCounter(&startTime);
#else
    clock_gettime(CLOCK_MONOTONIC, &startTime);
#endif
}

double Timer::stop()
{
#ifdef _WIN32
    LARGE_INTEGER stopTime;
    QueryPerformanceCounter(&stopTime);
    return ((double)(stopTime.QuadPart - startTime.QuadPart) / (double)frequency.QuadPart) - overhead;
#else
    timespec stopTime;
    clock_gettime(CLOCK_MONOTONIC, &stopTime);
    return (double)(stopTime.tv_sec - startTime.tv_sec) + (double)(stopTime.tv_nsec - startTime.tv_nsec) / 1000000000.0;
#endif
}
