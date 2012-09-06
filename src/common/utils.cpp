#include <sstream>
#include <iomanip>


#include "utils.h"

using namespace std;

uint32_t pow2roundup(uint32_t x)
{
    if (x < 0)
        return 0;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;

    return x + 1;
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
