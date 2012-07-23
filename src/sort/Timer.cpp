#include <iostream>

#include "timer.h"

using namespace std;

Timer::Timer()
{
    if (!QueryPerformanceFrequency(&frequency))
    {
        cerr << "High performance counter is not supported!" << endl;
        exit(0);
    }

    start();
    overhead = stop();
}

void Timer::start()
{
    QueryPerformanceCounter(&startTime);
}

double Timer::stop()
{
    LARGE_INTEGER stopTime;
    QueryPerformanceCounter(&stopTime);
    return ((double)(stopTime.QuadPart - startTime.QuadPart) / (double)frequency.QuadPart) - overhead;
}
