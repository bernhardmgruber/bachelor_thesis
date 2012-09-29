#ifndef CPUALGORITHM_H
#define CPUALGORITHM_H

#include <string>

using namespace std;

template <typename T>
class CPUAlgorithm
{
    public:
        virtual const string getName() = 0;
        virtual void run(T* data, T* result, size_t size) = 0;
};

#endif // CPUALGORITHM_H
