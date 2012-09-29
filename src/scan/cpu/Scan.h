#ifndef CPUSCAN_H
#define CPUSCAN_H

#include "../../common/CPUAlgorithm.h"
#include "../ScanAlgorithm.h"

namespace cpu
{
    template<typename T>
    class Scan : public CPUAlgorithm<T>, public ScanAlgorithm
    {
        public:
            const string getName() override
            {
                return "Scan (inclusiv)";
            }

            bool isInclusiv() override
            {
                return true;
            }

            void run(T* data, T* result, size_t size)
            {
                result[0] = data[0];
                for(size_t i = 1; i < size; i++)
                    result[i] = result[i - 1] + data[i];
            }

            virtual ~Scan() {}
    };
}

#endif // CPUSCAN_H
