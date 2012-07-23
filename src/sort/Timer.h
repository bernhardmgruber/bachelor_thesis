#ifndef TIMER_H
#define TIMER_H

#include <windows.h>

class Timer
{
    public:
        Timer();
        void start();
        double stop();


    private:
        LARGE_INTEGER frequency;
        LARGE_INTEGER startTime;
        double overhead;
};

#endif // TIMER_H
