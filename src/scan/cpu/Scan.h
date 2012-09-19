#ifndef CPUSCAN_H
#define CPUSCAN_H

#include "../../common/CPUAlgorithm.h"
#include "../ScanAlgorithm.h"

namespace cpu
{
    template<typename T, size_t count>
    class Scan : public CPUAlgorithm<T, count>, public ScanAlgorithm
    {
        public:
            string getName() override
            {
                return "Scan (inclusiv)";
            }

            bool isInclusiv() override
            {
                return true;
            }

            void run(T* data, T* result)
            {
                result[0] = data[0];
                for(size_t i = 1; i < count; i++)
                    result[i] = result[i - 1] + data[i];
            }

            virtual ~Scan() {}
    };
}

#endif // CPUSCAN_H
