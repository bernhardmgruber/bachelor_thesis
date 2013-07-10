#include <sstream>
#include <iomanip>
#include <stdlib.h>
#include <intrin.h>

#include "utils.h"

using namespace std;

size_t pow2roundup(size_t x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    if(sizeof(size_t) >= 8)
        x |= x >> 32;

    return x + 1;
}

size_t roundToMultiple(size_t x, size_t multiple)
{
    if(x % multiple == 0)
        return x;
    else
        return (x / multiple + 1) * multiple;
}

string sizeToString(size_t size)
{
    stringstream ss;

    if(size < 1024)
        ss <<  size << " B";
    else if(size < 1024 * 1024)
        ss << fixed << setprecision(1) << (size / 1024.0) << " KiB";
    else if(size < 1024 * 1024 * 1024)
        ss << fixed << setprecision(1) << (size / 1024.0 / 1024.0) << " MiB";
    else
        ss << fixed << setprecision(1) << (size / 1024.0 / 1024.0 / 1024.0) << " GiB";

    return ss.str();
}

string timeToString(double time)
{
    stringstream ss;

    size_t hours = (size_t)(time / 3600.0);
    if(hours > 0)
        ss << hours << "h ";
    time = fmod(time, 3600);

    size_t minutes = (size_t)(time / 60.0);
    if(minutes > 0)
        ss << minutes << "min ";
    time = fmod(time, 60);

    size_t seconds = (size_t) time;
    if(seconds > 0)
        ss << seconds << "s ";
    time = fmod(time, 1);

    size_t millis = (size_t)(time * 1000.0);
    ss << millis << "ms";

    return ss.str();
}

unsigned int ctz(unsigned int x)
{
#ifdef __GNUG__
    return __builtin_ctz(x);
#else
    unsigned long r;
    _BitScanForward(&r, x);
    return r;
#endif
}

unsigned int rootPowerOfTwo(unsigned int value, unsigned int root) {
    return 1 << (ctz(value) / root);
}

const string runTypeToString(const RunType runType)
{
    switch(runType)
    {
    case RunType::CPU:
        return "CPU";
    case RunType::CL_GPU:
        return "OpenCL GPU";
    case RunType::CL_CPU:
        return "OpenCL CPU";
    }

    throw exception("Invalid RunType");
}