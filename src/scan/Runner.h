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
        const int FLOAT_PRECISION = 3;

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
            printDevice(cpuContext);
            printDevice(gpuContext);
        }

        void printDevice(Context* context)
        {
            cout << "Device " << context->getInfo<string>(CL_DEVICE_NAME) << endl;
            cout << "   " << context->getInfo<string>(CL_DEVICE_VENDOR) << endl;
            cout << "   " << context->getInfo<size_t>(CL_DEVICE_MAX_COMPUTE_UNITS) << " compute units" << endl;
            cout << "   " << context->getInfo<size_t>(CL_DEVICE_MAX_CLOCK_FREQUENCY) << " MHz frequency" << endl;
            cout << "   " << fixed << setprecision(FLOAT_PRECISION) << sizeToString(context->getInfo<cl_ulong>(CL_DEVICE_GLOBAL_MEM_SIZE)) << " global mem" << endl;
            cout << "   " << fixed << setprecision(FLOAT_PRECISION) << sizeToString(context->getInfo<cl_ulong>(CL_DEVICE_LOCAL_MEM_SIZE)) << " local mem" << endl;
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
            cout << "#  Scan      " << fixed << setprecision(FLOAT_PRECISION) << scanTime << "s " << (verify(alg->isInclusiv()) ? "SUCCESS" : "FAILED ") << endl;

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
        struct Stats
        {
            double uploadTime;
            double runTime;
            double downloadTime;
            bool verificationResult;
            bool exceptionOccured;
            string exceptionMsg;

            Stats()
                : uploadTime(0), runTime(0), downloadTime(0), verificationResult(false), exceptionOccured(false)
            {
            }
        };

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

            map<int, Stats> stats;

            size_t maxWorkGroupSize = min(context->getInfo<size_t>(CL_DEVICE_MAX_WORK_GROUP_SIZE), count);
            if(!useMultipleWorkGroupSizes)
                stats[maxWorkGroupSize] = uploadRunDownload(alg, context, queue, maxWorkGroupSize);
            else
                for(size_t i = 1; i <= maxWorkGroupSize; i <<= 1)
                    stats[i] = uploadRunDownload(alg, context, queue, i);

            // cleanup
            timer.start();
            alg->cleanup();
            double cleanupTime = timer.stop();

            // calculate sum figures
            pair<int, Stats> fastest = *min_element(stats.begin(), stats.end(), [](pair<int, Stats> a, pair<int, Stats> b)
            {
                if(a.second.exceptionOccured || !a.second.verificationResult)
                    return false;
                if(b.second.exceptionOccured || !b.second.verificationResult)
                    return true;
                return a.second.uploadTime + a.second.runTime + a.second.downloadTime < b.second.uploadTime + b.second.runTime + b.second.downloadTime;
            });

            double uploadAvg = 0;
            for(auto e : stats)
                if(!e.second.exceptionOccured)
                    uploadAvg += e.second.uploadTime;
            uploadAvg /= stats.size();

            double downloadAvg = 0;
            for(auto e : stats)
                if(!e.second.exceptionOccured)
                    downloadAvg += e.second.downloadTime;
            downloadAvg /= stats.size();

            // print results
            cout << "#  (Init)         " << fixed << setprecision(FLOAT_PRECISION) << initTime << "s" << endl;
            cout << "#  Upload (avg)   " << fixed << setprecision(FLOAT_PRECISION) << uploadAvg << "s" << endl;

            for(auto s : stats)
                if(s.second.exceptionOccured)
                    cout << "#  Scan           EXCEPTION" << " (WG size: " << s.first << "): " << s.second.exceptionMsg << endl;
                else
                    cout << "#  Scan           " << fixed << setprecision(FLOAT_PRECISION) << s.second.runTime << "s " << "(WG size: " << s.first << ") " << (s.second.verificationResult ? "SUCCESS" : "FAILED ") << endl;

            cout << "#  Download (avg) " << fixed << setprecision(FLOAT_PRECISION) << downloadAvg << "s" << endl;
            cout << "#  (Cleanup)      " << fixed << setprecision(FLOAT_PRECISION) << cleanupTime << "s" << endl;
            cout << "#  Fastest        " << fixed << setprecision(FLOAT_PRECISION) << (fastest.second.uploadTime + fastest.second.runTime + fastest.second.downloadTime) << "s " << "(WG size: " << fastest.first << ") " << endl;

            // delete the algorithm and finish this test
            delete alg;

            finishTest();
        }

        template <template <typename, size_t> class A>
        inline Stats uploadRunDownload(A<T, count>* alg, Context* context, CommandQueue* queue, size_t workGroupSize)
        {
            Stats stats;

            try
            {
                // upload data
                timer.start();
                alg->upload(context, workGroupSize, data);
                stats.uploadTime = timer.stop();

                // run algorithm
                timer.start();
                alg->scan(queue, workGroupSize);
                stats.runTime = timer.stop();

                // download data
                timer.start();
                alg->download(queue, result);
                stats.downloadTime = timer.stop();

                // verify
                stats.verificationResult = verify(alg->isInclusiv());
            }
            catch(OpenCLException& e)
            {
                stats.exceptionOccured = true;
                stats.exceptionMsg = e.what();
            }

            return stats;
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
