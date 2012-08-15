#ifndef CPUPREFIXSUM_H
#define CPUPREFIXSUM_H

#include "../CPUScanAlgorithm.h"

namespace cpu
{
    template<typename T, size_t count>
    class PrefixSum : public CPUScanAlgorithm<T, count>
    {
        public:
            PrefixSum()
                : CPUScanAlgorithm<T, count>("Prefix Sum")
            {
            }

            void scan()
            {
                ScanAlgorithm<T, count>::scanResult[0] = ScanAlgorithm<T, count>::data[0];
                for(size_t i = 1; i < count; i++)
                    ScanAlgorithm<T, count>::scanResult[i] = ScanAlgorithm<T, count>::scanResult[i - 1] + ScanAlgorithm<T, count>::data[i];
            }
    };
}

#endif // CPUPREFIXSUM_H
