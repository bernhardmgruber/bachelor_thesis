#ifndef CPUSORTINGALGORITHM_H
#define CPUSORTINGALGORITHM_H

template<typename T, size_t count>
class CPUSortingAlgorithm
{
    public:
        virtual string getName() = 0;
        virtual void sort(T* data) = 0;
};

#endif // CPUSORTINGALGORITHM_H
