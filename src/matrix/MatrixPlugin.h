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
        string getTaskDescription(size_t size)
        {
            //return "Processing " << size << " elements of type " << getTypeName<T>() << " (" << sizeToString(size * sizeof(T)) << ")";
            stringstream ss;
            ss << "Multiplying " << size << "x" << size << " matrixes of type " << getTypeName<T>() << " (" << sizeToString(size * size * sizeof(T)) << " per matrix)";
            return ss.str();
        }

        void* genInput(size_t size)
        {
            size_t bufferSize = size * size * 2;

            T* data = new T[bufferSize]; // two size x size matrixes

            generate(data, data + bufferSize, [](){ return rand() % 100; });

            return data;
        }

        void* genResult(size_t size)
        {
            return new T[size * size];
        }

        void freeInput(void* data)
        {
            delete[] (T*)data;
        }

        void freeResult(void* result)
        {
            delete[] (T*)result;
        }

        bool verifyResult(MatrixAlgorithm* alg, void* data, void* result, size_t size)
        {
            return false;
        }
};

#endif // MATRIXPLUGIN_H
