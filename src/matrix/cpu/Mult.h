#ifndef CPUMULT_H
#define CPUMULT_H

#include "../../common/CPUAlgorithm.h"
#include "../MatrixAlgorithm.h"

namespace cpu
{
    template<typename T>
    class Mult : public CPUAlgorithm, public MatrixAlgorithm
    {
        public:
            const string getName() override
            {
                return "Matrix multiplication";
            }

            void run(void* data, void* result, size_t size) override
            {
                T* a = (T*)data;
                T* b = a + size * size;
                T* r = (T*) result;

                for(size_t i = 0; i < size; i++)
                {
                    for(size_t j = 0; j < size; j++)
                    {
                        r[i * size + j] = 0;
                        for(size_t k = 0; k < size; k++)
                            r[i * size + j] += a[i * size + k] * b[k * size + j];
                    }
                }
            }

            virtual ~Mult() {}

        private:
    };
}

#endif // CPUMULT_H
