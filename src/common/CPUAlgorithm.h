#ifndef CPUALGORITHM_H
#define CPUALGORITHM_H

#include <string>

using namespace std;

class CPUAlgorithm
{
    public:
        virtual string getName() = 0;
        virtual void run(void* data, void* result, size_t size) = 0;
};

#endif // CPUALGORITHM_H
