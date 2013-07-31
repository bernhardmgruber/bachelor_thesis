#pragma once

#include <sstream>

#include "ScanAlgorithm.h"

using namespace std;

template <typename T>
class ScanPlugin
{
    public:
        typedef ScanAlgorithm AlgorithmType;

        const string getTaskDescription(size_t size)
        {
            stringstream ss;
            ss << "Scanning " << size << " elements of type " << getTypeName<T>() << " (" << sizeToString(size * sizeof(T)) << ")";
            return ss.str();
        }

        T* genInput(size_t size)
        {
            T* data = new T[size];

            generate(data, data + size, []()
            {
                return rand() % 100;
                //return 1;
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

        bool verifyResult(ScanAlgorithm* alg, T* data, T* result, size_t size)
        {
            if(alg->isInclusiv())
            {
                if(data[0] != result[0])
                    return false;

                for(size_t i = 1; i < size; i++)
                    if(result[i] != result[i - 1] + data[i])
                        return false;

                return true;
            }
            else
            {
                if(result[0] != 0)
                    return false;

                for(size_t i = 1; i < size; i++)
                    if(result[i] != result[i - 1] + data[i - 1])
                        return false;

                return true;
            }
        }
};
