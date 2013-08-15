#pragma once

#include <sstream>
#include <random>

#include "SortAlgorithm.h"

//#define FULL_VERIFY

using namespace std;

template <typename T>
class SortPlugin
{
public:
    typedef SortAlgorithm AlgorithmType;

    const string getTaskDescription(size_t size)
    {
        stringstream ss;
        ss << "Sorting " << size << " elements of type " << getTypeName<T>() << " (" << sizeToString(size * sizeof(T)) << ")";
        return ss.str();
    }

    T* genInput(size_t size)
    {
        T* data = new T[size]; // two size x size matrixes

        default_random_engine generator;
        uniform_int_distribution<T> dist; // range is [0;numeric_limits<T>::max()]

        generate(data, data + size, [&]()
        {
            return rand() % 100;
            //return (rand() % 256) << 24;
            //return dist(generator);
        });

        return data;
    }

    T* genResult(size_t size)
    {
        return new T[size];
    }

    void freeInput(T* data)
    {
        delete[] (T*)data;
    }

    void freeResult(T* result)
    {
        delete[] (T*)result;
    }

    bool verifyResult(SortAlgorithm* alg, T* data, T* result, size_t size)
    {
#ifdef FULL_VERIFY
        if(!alg->isInPlace())
        {
            // full test
            sort(data, data + size);
            return memcmp(data, result, size * sizeof(T)) == 0;
        }
        else
        {
            for(size_t i = 0; i < size - 1; i++)
                if(result[i] > result[i + 1])
                    return false;
            return true;
        }
#else
        if(alg->isInPlace())
            result = data;

        for(size_t i = 0; i < size - 1; i++)
            if(result[i] > result[i + 1])
                return false;
        return true;
#endif
    }
};