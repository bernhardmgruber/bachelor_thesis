#ifndef CPUSCANALGORITHM_H
#define CPUSCANALGORITHM_H

template<typename T, size_t count>
class CPUScanAlgorithm
{
    public:
        virtual string getName() = 0;
        virtual bool isInclusiv() = 0;
        virtual void scan(T* data, T* result) = 0;
};


#endif // CPUSCANALGORITHM_H
