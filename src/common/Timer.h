#pragma once

#ifdef _WIN32
#include <windows.h>
#undef min
#undef max
#else
#include <time.h>
#endif

class Timer
{
    public:
        Timer();
        void start();
        double stop();

    private:
        #ifdef _WIN32
        LARGE_INTEGER frequency;
        LARGE_INTEGER startTime;
        #else
        //timespec frequency;
        timespec startTime;
        #endif
        double overhead;
};
