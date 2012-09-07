#ifndef RUNNER_H
#define RUNNER_H

#include "../common/OpenCL.h"
#include "../common/Timer.h"
#include "../common/utils.h"

#include <typeinfo>
#include <iomanip>
#include <map>
#include <algorithm>

using namespace std;

/**
 * Used to run scan algorithms.
 */
template <typename T, size_t count>
class Runner
{
    public:
        Runner()
        {
            OpenCL::init();
            gpuContext = OpenCL::getGPUContext();
            cpuContext = OpenCL::getCPUContext();
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

        /**
         * Prints some information about the OpenCL devices used.
         */
        void printCLInfo()
        {
            cout << "Running on CPU " << cpuContext->getInfoString(CL_DEVICE_NAME) << endl;
            cout << "   " << fixed << setprecision(2) << sizeToString(cpuContext->getInfoSize(CL_DEVICE_GLOBAL_MEM_SIZE)) << " global mem" << endl;
            cout << "   " << fixed << setprecision(2) << sizeToString(cpuContext->getInfoSize(CL_DEVICE_LOCAL_MEM_SIZE)) << " local mem" << endl;
            cout << endl;
            cout << "Running on GPU " << gpuContext->getInfoString(CL_DEVICE_NAME) << endl;
            cout << "   " << fixed << setprecision(2) << sizeToString(gpuContext->getInfoSize(CL_DEVICE_GLOBAL_MEM_SIZE)) << " global mem" << endl;
            cout << "   " << fixed << setprecision(2) << sizeToString(gpuContext->getInfoSize(CL_DEVICE_LOCAL_MEM_SIZE)) << " local mem" << endl;
            cout << endl;
        }

        /**
         * Runs an algorithm on the CPU.
         */
        template <template <typename, size_t> class A>
        void run()
        {
            // create a new instance of our test algorithm and prepare the test run
            A<T, count>* alg = new A<T, count>();
            prepareTest(alg->getName());

            // run algorithms
            timer.start();
            alg->scan(data, result);
            double scanTime = timer.stop();

            // print results
            cout << "#  Scan      " << fixed << scanTime << "s" << flush << endl;
            cout << "#  " << (verify(alg->isInclusiv()) ? "SUCCESS" : "FAILED ") << "   " << fixed << scanTime << "s" << flush << endl;

            // delete the algorithm and finish this test
            delete alg;

            finishTest();
        }

        /**
         * Runs an algorithm on the CPU using OpenCL.
         */
        template <template <typename, size_t> class A>
        void runCLCPU(bool useMultipleWorkGroupSizes)
        {
            runCL<A>(cpuContext, cpuQueue, useMultipleWorkGroupSizes);
        }

        /**
         * Runs an algorithm on the GPU using OpenCL.
         */
        template <template <typename, size_t> class A>
        void runCLGPU(bool useMultipleWorkGroupSizes)
        {
            runCL<A>(gpuContext, gpuQueue, useMultipleWorkGroupSizes);
        }

    private:
        template <template <typename, size_t> class A>
        void runCL(Context* context, CommandQueue* queue, bool useMultipleWorkGroupSizes)
        {
            // create a new instance of our test algorithm and prepare the test run
            A<T, count>* alg = new A<T, count>();
            prepareTest(alg->getName());

            // run custom initialization
            timer.start();
            alg->init(context);
            double initTime = timer.stop();

            // upload data
            timer.start();
            alg->upload(context, data);
            double uploadTime = timer.stop();

            // run sorting algorithm
            size_t maxWorkGroupSize = min(context->getInfoSize(CL_DEVICE_MAX_WORK_GROUP_SIZE), count);
            map<int, double> scanTimes;
            if(!useMultipleWorkGroupSizes)
            {
                timer.start();
                alg->scan(queue, maxWorkGroupSize);
                scanTimes[maxWorkGroupSize] = timer.stop();
            }
            else
            {
                for(size_t i = 1; i <= maxWorkGroupSize; i <<= 1)
                {
                    // check if work group size divides the input
                    if(count % i == 0)
                    {
                        timer.start();
                        alg->scan(queue, i);
                        scanTimes[i] = timer.stop();
                    }
                }
            }

            // download data
            timer.start();
            alg->download(queue, result);
            double downloadTime = timer.stop();

            // cleanup
            timer.start();
            alg->cleanup();
            double cleanupTime = timer.stop();

            // print results

            cout << "#  (Init)    " << fixed << initTime << "s" << flush << endl;
            cout << "#  Upload    " << fixed << uploadTime << "s" << flush << endl;

            for(auto entry : scanTimes)
                cout << "#  Scan      " << fixed << entry.second << "s " << "(WG size: " << entry.first << ")" << flush << endl;

            cout << "#  Download  " << fixed << downloadTime << "s" << flush << endl;
            cout << "#  Cleanup   " << fixed << cleanupTime << "s" << flush << endl;
            cout << "#  " << (verify(alg->isInclusiv()) ? "SUCCESS" : "FAILED ") << "   " << fixed << (/*initTime +*/ uploadTime + min_element(scanTimes.begin(), scanTimes.end(), [](pair<int, double> a, pair<int, double> b)
            {
                return a.second < b.second;
            })->second + downloadTime + cleanupTime) << "s (fastest)" << flush << endl;

            // delete the algorithm and finish this test
            delete alg;

            finishTest();
        }

        void prepareTest(string name)
        {
            cout << "###############################################################################" << endl;
            cout << "# " << name << endl;
            cout << "#  Scaning " << count << " elements of type " << typeid(T).name() << " (" << sizeToString(count * sizeof(T)) << ")" << endl;

            // generate random array
            data = new T[count];
            for(size_t i = 0; i < count; i++)
                data[i] = 1; //rand() % 100;
            // allocate storage for the result
            result = new T[count];
        }

        void finishTest()
        {
            delete[] data;
            delete[] result;

            cout << "###############################################################################" << endl;
            cout << endl;
        }

        bool verify(bool inclusiv)
        {
            if(inclusiv)
            {
                if(data[0] != result[0])
                    return false;

                for(size_t i = 1; i < count; i++)
                    if(result[i] !=  result[i - 1] + data[i])
                        return false;

                return true;
            }
            else
            {
                if(result[0] != 0)
                    return false;

                for(size_t i = 1; i < count; i++)
                    if(result[i] != result[i - 1] + data[i - 1])
                        return false;

                return true;
            }
        }

        Context* gpuContext;
        Context* cpuContext;
        CommandQueue* gpuQueue;
        CommandQueue* cpuQueue;

        Timer timer;

        T* data;
        T* result;
};

#endif // RUNNER_H
