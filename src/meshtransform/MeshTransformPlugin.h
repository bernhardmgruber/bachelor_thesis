#ifndef MESHTRANSFORMPLUGIN_H
#define MESHTRANSFORMPLUGIN_H

#include <sstream>

#include "MeshTransformAlgorithm.h"

using namespace std;

template <typename T>
class MeshTransformPlugin
{
    public:
        typedef MeshTransformAlgorithm AlgorithmType;

        const string getTaskDescription(size_t size)
        {
            stringstream ss;
            ss << "Transforming mesh with " << size << " vertices (" << sizeToString(size * sizeof(T)) << ")";
            return ss.str();
        }

        T* genInput(size_t size)
        {
            T* data = new T[size];

            generate(data, data + size, []()
            {
                return rand() % 100;
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

        bool verifyResult(MeshTransformAlgorithm* alg, T* data, T* result, size_t size)
        {
            return true;
        }
};

#endif // MESHTRANSFORMPLUGIN_H
