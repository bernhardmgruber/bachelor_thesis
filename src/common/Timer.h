#ifndef TIMER_H
#define TIMER_H

#ifdef _WIN32
#include <windows.h>
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

#endif // TIMER_H
