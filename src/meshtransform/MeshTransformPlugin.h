#ifndef MESHTRANSFORMPLUGIN_H
#define MESHTRANSFORMPLUGIN_H

#include <string.h>
#include <sstream>

#include "MeshTransformAlgorithm.h"

using namespace std;

template <typename T>
class MeshTransformPlugin
{
    public:
        typedef MeshTransformAlgorithm AlgorithmType;

        static const size_t MATRIX_SIZE = MeshTransformAlgorithm::MATRIX_SIZE;

        const string getTaskDescription(size_t size)
        {
            stringstream ss;
            ss << "Transforming mesh with " << size << " vertices (" << sizeToString(size * 3 * sizeof(T)) << ")";
            return ss.str();
        }

        T* genInput(size_t size)
        {
            T* data = new T[MATRIX_SIZE + size * 3];

            /*T matrix[] = {1, 0, 0, 0,
                          0, 1, 0, 0,
                          0, 0, 1, 0,
                          0, 0, 0, 1};

            memcpy(data, matrix, MATRIX_SIZE * sizeof(T));*/

            generate(data, data + MATRIX_SIZE + size * 3, []()
            {
                return rand() % 10;
            });

            return data;
        }

        T* genResult(size_t size)
        {
            return new T[size * 3];
        }

        void freeInput(T* data)
        {
            delete[] (T*)data;
        }

        void freeResult(T* result)
        {
            delete[] (T*)result;
        }

        bool verifyResult(MeshTransformAlgorithm* alg, T* data, T* result, size_t size)
        {
            T* matrix = data;
            T* vectors = data + MATRIX_SIZE;

            for(size_t i = 0; i < size; i++)
            {
                T r[3];
                matrixMultiplication(matrix, &vectors[i * 3], r);

                if(memcmp(r, &result[i * 3], sizeof(T) * 3) != 0)
                    return false;
            }

            return true;
        }

        inline void matrixMultiplication(T* m, T* v, T* r)
        {
            r[0] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2] + m[3];
            r[1] = m[4] * v[0] + m[5] * v[1] + m[6] * v[2] + m[7];
            r[2] = m[8] * v[0] + m[9] * v[1] + m[10] * v[2] + m[11];
        }
};

#endif // MESHTRANSFORMPLUGIN_H
