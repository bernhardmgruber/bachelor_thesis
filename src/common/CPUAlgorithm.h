#ifndef CPUALGORITHM_H
#define CPUALGORITHM_H

template<typename T, size_t count>
class CPUAlgorithm
{
    public:
        virtual string getName() = 0;
        virtual void run(T* data, T* result) = 0;
};

#endif // CPUALGORITHM_H
