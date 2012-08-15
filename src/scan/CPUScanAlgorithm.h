#ifndef CPUSCANALGORITHM_H
#define CPUSCANALGORITHM_H

#include "ScanAlgorithm.h"

template<typename T, size_t count>
class CPUScanAlgorithm : public ScanAlgorithm<T, count>
{
    using Base = CPUScanAlgorithm<T, count>;

 public:
        CPUScanAlgorithm(string name)
            : ScanAlgorithm<T, count>(name)
        {
        }

        virtual ~CPUScanAlgorithm()
        {
        }

        void runStages()
        {
                // run sorting algorithm
                Base::timer.start();
                scan();
                double scanTime = Base::timer.stop();

                cout << "#  Scan      " << fixed << scanTime << "s" << flush << endl;
                cout << "#  " << (Base::verify() ? "SUCCESS" : "FAILED ") << "   " << fixed << scanTime << "s" << flush << endl;
        }

    protected:
        virtual void scan() = 0;

};

template <template <typename, size_t> class T, size_t count, typename V>
void runCPU()
{
    ScanAlgorithm<V, count>* alg;
    alg = new T<V, count>();
    alg->runTest();
    delete alg;
}

#define RUN(algorithmTestClass, count, valueType) runCPU<algorithmTestClass, count, valueType>();

#endif // CPUSCANALGORITHM_H
