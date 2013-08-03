#pragma once

#include "../../common/CPUAlgorithm.h"
#include "../ScanAlgorithm.h"

namespace cpu
{
    template<typename T>
    class Scan : public CPUAlgorithm<T>, public ScanAlgorithm
    {
        static const bool INCLUSIVE = false;

    public:
        const string getName() override
        {
            return string("Scan ") + (INCLUSIVE ? "(inclusiv)" : "(exclusive)");
        }

        bool isInclusiv() override
        {
            return INCLUSIVE;
        }

        void run(T* data, T* result, size_t size)
        {
            if(INCLUSIVE)
            {
                result[0] = data[0];
                for(size_t i = 1; i < size; i++)
                    result[i] = result[i - 1] + data[i];
            }
            else
            {
                result[0] = 0;
                for(size_t i = 1; i < size; i++)
                    result[i] = result[i - 1] + data[i - 1];
            }
        }

        virtual ~Scan() {}
    };
}
