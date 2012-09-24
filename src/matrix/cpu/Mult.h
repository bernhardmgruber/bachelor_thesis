#ifndef CPUMULT_H
#define CPUMULT_H

#include "../../common/CPUAlgorithm.h"
#include "../MatrixAlgorithm.h"

namespace cpu
{
    template<typename T, size_t count>
    class Mult : public CPUAlgorithm<T, count>, public MatrixAlgorithm
    {
        public:
            string getName() override
            {
                return "Matrix multiplication";
            }

            void run(T* data, T* result)
            {

            }

            virtual ~Mult() {}

        private:
    };
}

#endif // CPUMULT_H
