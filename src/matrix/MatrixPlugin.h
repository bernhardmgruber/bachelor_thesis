#ifndef MATRIXPLUGIN_H
#define MATRIXPLUGIN_H

#include <stdlib.h>
#include <sstream>

#include "MatrixAlgorithm.h"

using namespace std;

template <typename T>
class MatrixPlugin
{
    public:
        typedef MatrixAlgorithm AlgorithmType;

        const string getTaskDescription(size_t size)
        {
            //return "Processing " << size << " elements of type " << getTypeName<T>() << " (" << sizeToString(size * sizeof(T)) << ")";
            stringstream ss;
            ss << "Multiplying " << size << "x" << size << " matrixes of type " << getTypeName<T>() << " (" << sizeToString(size * size * sizeof(T)) << " per matrix)";
            return ss.str();
        }

        T* genInput(size_t size)
        {
            size_t bufferSize = size * size * 2;

            T* data = new T[bufferSize]; // two size x size matrixes

            generate(data, data + bufferSize, []() -> T
            {
                return rand() % 100;
            });

            return data;
        }

        T* genResult(size_t size)
        {
            return new T[size * size];
        }

        void freeInput(T* data)
        {
            delete[] (T*)data;
        }

        void freeResult(T* result)
        {
            delete[] (T*)result;
        }

        bool verifyResult(MatrixAlgorithm* alg, T* data, T* result, size_t size)
        {
            T* a = (T*)data;
            T* b = a + size * size;
            T* c = (T*) result;

            bool success = true;

            #pragma omp parallel for
            for(size_t i = 0; i < size; i++)
            {
                if(success)
                    for(size_t j = 0; j < size; j++)
                    {
                        T sum = 0;
                        for(size_t k = 0; k < size; k++)
                            sum += a[i * size + k] * b[k * size + j];
                        if(c[i * size + j] != sum)
                            success = false;
                    }
            }

            return success;

            /*for(size_t i = 0; i < size; i++)
            {
                for(size_t j = 0; j < size; j++)
                {
                    T sum = 0;
                    for(size_t k = 0; k < size; k++)
                        sum += a[i * size + k] * b[k * size + j];
                    if(c[i * size + j] != sum)
                    {
                        printArr(c, size * size);
                        return false;
                    }
                }
            }

            return true;*/
        }
};

#endif // MATRIXPLUGIN_H
