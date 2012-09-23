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
 * Used to run algorithms.
 */
template <typename T, size_t count, template <typename, size_t, template <typename, size_t> class> class V>
class Runner
{
    public:
        const int FLOAT_PRECISION = 3;

        Runner(bool noCPU = true)
            : noCPU(noCPU)
        {
            OpenCL::init();
            gpuContext = OpenCL::getGPUContext();
            if(noCPU)
                cpuContext = nullptr;
            else
                cpuContext = OpenCL::getCPUContext();
            gpuQueue = gpuContext->createCommandQueue();
            if(cpuContext)
                cpuQueue = cpuContext->createCommandQueue();
        }

        virtual ~Runner()
        {
            delete gpuContext;
            delete gpuQueue;
            if(cpuContext)
            {
                delete cpuContext;
                delete cpuQueue;
            }
            OpenCL::cleanup();
        }

        /**
         * Prints some information about the OpenCL devices used.
         */
        void printCLInfo()
        {
            cout << "Running on the following devices:" << endl;
            if(cpuContext)
                printDevice(cpuContext);
            printDevice(gpuContext);
            cout << endl;
        }

        void printDevice(Context* context)
        {
            cout << "" << context->getInfo<string>(CL_DEVICE_NAME) << endl;
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
            alg->run(data, result);
            double scanTime = timer.stop();

            // print results
            cout << "#  Run       " << fixed << setprecision(FLOAT_PRECISION) << scanTime << "s " << (V<T, count, A>::verify(alg, data, result) ? "SUCCESS" : "FAILED ") << endl;

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
                    cout << "#  Run            EXCEPTION" << " (WG size: " << s.first << "): " << s.second.exceptionMsg << endl;
                else
                    cout << "#  Run            " << fixed << setprecision(FLOAT_PRECISION) << s.second.runTime << "s " << "(WG size: " << s.first << ") " << (s.second.verificationResult ? "SUCCESS" : "FAILED ") << endl;

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
                alg->upload(context, queue, workGroupSize, data);
                queue->finish();
                stats.uploadTime = timer.stop();

                // run algorithm
                timer.start();
                alg->run(queue, workGroupSize);
                queue->finish();
                stats.runTime = timer.stop();

                // download data
                timer.start();
                alg->download(queue, result);
                queue->finish();
                stats.downloadTime = timer.stop();

                // verify
                stats.verificationResult = V<T, count, A>::verify(alg, data, result);
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
            cout << "#  Processing " << count << " elements of type " << getTypeName<T>() << " (" << sizeToString(count * sizeof(T)) << ")" << endl;

            // generate random array
            data = new T[count];
            for(size_t i = 0; i < count; i++)
                data[i] = rand() % 100;
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

        Context* gpuContext;
        Context* cpuContext;
        CommandQueue* gpuQueue;
        CommandQueue* cpuQueue;

        Timer timer;

        T* data;
        T* result;

        bool noCPU;
};

#endif // RUNNER_H
