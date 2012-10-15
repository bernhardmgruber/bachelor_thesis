#ifndef CPUCBLASMULT_H
#define CPUCBLASMULT_H

#include "../../../common/libs/cblas/include/cblas.h"
#include "../../../common/CPUAlgorithm.h"
#include "../../MatrixAlgorithm.h"

namespace cpu
{
    namespace cblas
    {
        template<typename T>
        class Mult : public CPUAlgorithm<T>, public MatrixAlgorithm
        {
            public:
                const string getName() override
                {
                    return "Matrix multiplication (CBLAS)";
                }

                void run(T* data, T* result, size_t size) override
                {
                    throw invalid_argument("template type is not supported");
                }

                virtual ~Mult() {}

            private:
        };

        template <>
        void Mult<float>::run(float* data, float* result, size_t size)
        {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, size,  size,  size, 1.0, data, size, data + size * size, size, 0.0, result, size);
        }
    }
}

#endif // CPUCBLASMULT_H
