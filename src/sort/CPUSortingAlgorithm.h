#ifndef CPUSORTINGALGORITHM_H
#define CPUSORTINGALGORITHM_H

#include "SortingAlgorithm.h"

template<typename T, size_t count>
class CPUSortingAlgorithm : public SortingAlgorithm<T, count>
{
    using Base = SortingAlgorithm<T, count>;

 public:
        CPUSortingAlgorithm(string name)
            : SortingAlgorithm<T, count>(name)
        {
        }

        virtual ~CPUSortingAlgorithm()
        {
        }

        void runStages()
        {
                // run sorting algorithm
                Base::timer.start();
                sort();
                double sortTime = Base::timer.stop();

                cout << "#  Sort      " << fixed << sortTime << "s" << flush << endl;
                cout << "#  " << (Base::isSorted() ? "SUCCESS" : "FAILED ") << "   " << fixed << sortTime << "s" << flush << endl;
        }

    protected:
        virtual void sort() = 0;

};

template <typename T>
inline void swap(T& a, T& b)
{
    T tmp = a;
    a = b;
    b = tmp;
}


template <template <typename, size_t> class T, size_t count, typename V>
void runCPU()
{
    SortingAlgorithm<V, count>* alg;
    alg = new T<V, count>();
    alg->runTest();
    delete alg;
}

#define RUN(algorithmTestClass, count, valueType) runCPU<algorithmTestClass, count, valueType>();

#endif // CPUSORTINGALGORITHM_H
