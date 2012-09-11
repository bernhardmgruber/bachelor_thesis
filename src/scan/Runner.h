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

            map<int, double> uploadTimes;
            map<int, double> scanTimes;
            map<int, double> downloadTimes;
            map<int, bool> verifications;

            size_t maxWorkGroupSize = min(context->getInfoSize(CL_DEVICE_MAX_WORK_GROUP_SIZE), count);
            if(!useMultipleWorkGroupSizes)
            {
                // upload data
                timer.start();
                alg->upload(context, maxWorkGroupSize, data);
                uploadTimes[maxWorkGroupSize] = timer.stop();

                // run algorithm
                timer.start();
                alg->scan(queue, maxWorkGroupSize);
                scanTimes[maxWorkGroupSize] = timer.stop();

                // download data
                timer.start();
                alg->download(queue, result);
                downloadTimes[maxWorkGroupSize] = timer.stop();

                // verify
                verifications[maxWorkGroupSize] = verify(alg->isInclusiv());
            }
            else
            {
                for(size_t i = 1; i <= maxWorkGroupSize; i <<= 1)
                {
                    // upload data
                    timer.start();
                    alg->upload(context, i, data);
                    uploadTimes[i] = timer.stop();

                    // run algorithm
                    timer.start();
                    alg->scan(queue, i);
                    scanTimes[i] = timer.stop();

                    // download data
                    timer.start();
                    alg->download(queue, result);
                    downloadTimes[i] = timer.stop();

                    // verify
                    verifications[i] = verify(alg->isInclusiv());
                }
            }

            // cleanup
            timer.start();
            alg->cleanup();
            double cleanupTime = timer.stop();

            // calculate sum figures
            map<int, double> runs;
            for(size_t i = 1; i <= maxWorkGroupSize; i <<= 1)
                runs[i] = uploadTimes[i] + scanTimes[i] + downloadTimes[i];
            pair<int, double> fastest = *min_element(runs.begin(), runs.end(), [](pair<int, double> a, pair<int, double> b)
            {
                return a.second < b.second;
            });

            double uploadAvg = 0;
            for(auto e : uploadTimes)
                uploadAvg += e.second;
            uploadAvg /= uploadTimes.size();

            double downloadAvg = 0;
            for(auto e : downloadTimes)
                downloadAvg += e.second;
            downloadAvg /= downloadTimes.size();

            // print results
            cout << "#  (Init)         " << fixed << initTime << "s" << flush << endl;
            cout << "#  Upload (avg)   " << fixed << uploadAvg << "s" << flush << endl;

            for(size_t i = 1; i <= maxWorkGroupSize; i <<= 1)
                cout << "#  Scan           " << fixed << scanTimes[i] << "s " << "(WG size: " << i << ") " << (verifications[i] ? "SUCCESS" : "FAILED ") << flush << endl;

            cout << "#  Download (avg) " << fixed << downloadAvg << "s" << flush << endl;
            cout << "#  (Cleanup)      " << fixed << cleanupTime << "s" << flush << endl;
            cout << "#  Fastest        " << fixed << fastest.second << "s " << "(WG size: " << fastest.first << ") " << flush << endl;

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
