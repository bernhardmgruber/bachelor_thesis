#ifndef RUNNER_H
#define RUNNER_H

#include "../common/OpenCL.h"
#include "../common/Timer.h"

#include <typeinfo>

/**
 * Used to run sorting algorithms.
 */
template <typename T, size_t count>
class Runner
{
    public:
        Runner()
        {
            OpenCL::init();
            gpuContext = OpenCL::getGPUContext();
            cpuContext = OpenCL::getGPUContext();
            gpuQueue = gpuContext->createCommandQueue();
            cpuQueue = cpuContext->createCommandQueue();
        }

        virtual ~Runner()
        {
            delete gpuContext;
            delete cpuContext;
            delete gpuQueue;
            delete cpuQueue;
            OpenCL::cleanup();
        }

        template <template <typename, size_t> class A>
        void run()
        {
            // create a new instance of our test algorithm and prepare the test run
            A<T, count>* alg = new A<T, count>();

            prepareTest(alg->getName());

            // run algorithms
            timer.start();
            alg->sort(data);
            double sortTime = timer.stop();

            // delete the algorithm and finish this test
            delete alg;

            cout << "#  Sort      " << fixed << sortTime << "s" << flush << endl;
            cout << "#  " << (isSorted() ? "SUCCESS" : "FAILED ") << "   " << fixed << sortTime << "s" << flush << endl;

            finishTest();
        }

    private:
        void prepareTest(string name)
        {
            cout << "###############################################################################" << endl;
            cout << "# " << name << endl;
            cout << "#  Sorting " << count << " elements of type " << typeid(T).name() << " (" << ((sizeof(T) * count) >> 10) << " KiB)" << endl;

            // generate random array
            data = new T[count];
            for(size_t i = 0; i < count; i++)
                data[i] = rand();
        }

        void finishTest()
        {
            delete[] data;

            cout << "###############################################################################" << endl;
            cout << endl;
        }

        bool isSorted()
        {
            bool sorted = true;
            for(size_t i = 0; i < count - 1; i++)
                if(data[i] > data[i + 1])
                {
                    sorted = false;
                    break;
                }

            return sorted;
        }

        Context* gpuContext;
        Context* cpuContext;
        CommandQueue* gpuQueue;
        CommandQueue* cpuQueue;

        Timer timer;

        T* data;
};

#endif // RUNNER_H
