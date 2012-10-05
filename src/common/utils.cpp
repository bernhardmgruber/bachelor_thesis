#include <sstream>
#include <iomanip>
#include <stdlib.h>

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
    ldiv_t result = div((long)x, (long)multiple);

    if(result.rem == 0)
        return x;
    else
        return (result.quot + 1) * multiple;
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
