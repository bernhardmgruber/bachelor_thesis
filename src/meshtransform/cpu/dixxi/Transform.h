#ifndef CPUDIXXITRANSFORM_H
#define CPUDIXXITRANSFORM_H

#include "../../../common/CPUAlgorithm.h"
#include "../../MeshTransformAlgorithm.h"

namespace cpu
{
    namespace dixxi
    {
        template <typename T>
        class Transform : public CPUAlgorithm<T>, public MeshTransformAlgorithm
        {
            public:
                const string getName() override
                {
                    return "Transform";
                }

                void run(T* data, T* result, size_t size) override
                {
                    T* matrix = data;
                    T* vectors = data + MATRIX_SIZE;

                    for(size_t i = 0; i < size; i++)
                        matrixMultiplication(matrix, &vectors[i * 3], &result[i * 3]);
                }

                virtual ~Transform() {}

            private:
                inline void matrixMultiplication(T* m, T* v, T* r)
                {
                    r[0] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2] + m[3];
                    r[1] = m[4] * v[0] + m[5] * v[1] + m[6] * v[2] + m[7];
                    r[2] = m[8] * v[0] + m[9] * v[1] + m[10] * v[2] + m[11];
                }
        };
    }
}

#endif // CPUDIXXITRANSFORM_H
