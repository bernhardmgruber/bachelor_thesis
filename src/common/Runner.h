#ifndef RUNNER_H
#define RUNNER_H

#include "../common/OpenCL.h"
#include "../common/Timer.h"
#include "../common/utils.h"

#include <typeinfo>
#include <iomanip>
#include <map>
#include <algorithm>
#include <vector>
#include <string>

using namespace std;

/**
 * Used to run algorithms.
 */
template <typename T, template <typename> class Plugin>
class Runner
{
    public:
        const int FLOAT_PRECISION = 3;

        /**
         * Constructor
         */
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

            plugin = new Plugin<T>();
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

            delete plugin;

            for(Stats* s : stats)
                delete s;
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
         * Runs an algorithm on the CPU with the given problem size.
         * The results of the run are printed to stdout.
         */
        template <template <typename> class Algorithm>
        void printRun(size_t size)
        {
            CPURun* cpuRun = run<Algorithm>(size);

            cout << "###############################################################################" << endl;
            cout << "# " << cpuRun->algorithmName << endl;
            cout << "#  " << cpuRun->taskDescription << endl;
            cout << "#  CPU       " << fixed << setprecision(FLOAT_PRECISION) << cpuRun->runTime << "s " << (cpuRun->verificationResult ? "SUCCESS" : "FAILED") << endl;
            cout << "###############################################################################" << endl;
            cout << endl;

            stats.push_back(cpuRun);
        }

        /**
         * Runs an algorithm on the CPU using OpenCL.
         * The results of the run are printed to stdout.
         */
        template <template <typename> class Algorithm>
        void printRunCLCPU(size_t size, bool useMultipleWorkGroupSizes)
        {
            if(cpuContext)
                printRunCL<Algorithm>(cpuContext, cpuQueue, useMultipleWorkGroupSizes, size);
            else
                cout << "Cant run algorithm. No CPU context!" << endl;
        }

        /**
         * Runs an algorithm on the GPU using OpenCL.
         * The results of the run are printed to stdout.
         */
        template <template <typename> class Algorithm>
        void printRunCLGPU(size_t size, bool useMultipleWorkGroupSizes)
        {
            printRunCL<Algorithm>(gpuContext, gpuQueue, size, useMultipleWorkGroupSizes);
        }

    private:
        enum RunType
        {
            CPU,
            CL_CPU,
            CL_GPU
        };

        struct Stats
        {
            RunType runType;
            const string algorithmName;
            const string taskDescription;

            Stats(RunType runType, string algorithmName, string taskDescription)
                : runType(runType), algorithmName(algorithmName), taskDescription(taskDescription)
            {
            }
        };

        struct CPURun : public Stats
        {
            double runTime;
            bool verificationResult;
            bool exceptionOccured = false;
            string exceptionMsg;

            CPURun(RunType runType, string algorithmName, string taskDescription)
                : Stats(runType,algorithmName, taskDescription)
            {
            }
        };

        struct CLRun
        {
            size_t wgSize;
            double uploadTime;
            double runTime;
            double downloadTime;
            bool verificationResult;
            bool exceptionOccured = false;
            string exceptionMsg;
        };

        struct CLBatch : public Stats
        {
            double initTime;
            double cleanupTime;
            vector<CLRun> runs;
            typename vector<CLRun>::iterator fastest;
            double avgUploadTime;
            double avgRunTime;
            double avgDownloadTime;

            CLBatch(RunType runType, string algorithmName, string taskDescription)
                : Stats(runType,algorithmName, taskDescription)
            {
            }
        };

        template <template <typename> class Algorithm>
        void printRunCL(Context* context, CommandQueue* queue, size_t size, bool useMultipleWorkGroupSizes)
        {
            CLBatch* batch = runCL<Algorithm>(context, queue, size, useMultipleWorkGroupSizes);

            // print results
            cout << "###############################################################################" << endl;
            cout << "# " << batch->algorithmName << endl;
            cout << "#  " << batch->taskDescription << endl;
            cout << "#  (Init)         " << fixed << setprecision(FLOAT_PRECISION) << batch->initTime << "s" << endl;
            cout << "#  Upload (avg)   " << fixed << setprecision(FLOAT_PRECISION) << batch->avgUploadTime << "s" << endl;

            for(auto r : batch->runs)
                if(r.exceptionOccured)
                    cout << "#  GPU (WG: " << setw(4) << r.wgSize << ") EXCEPTION: " << r.exceptionMsg << endl;
                else
                    cout << "#  GPU (WG: " << setw(4) << r.wgSize << ") " << fixed << setprecision(FLOAT_PRECISION) << r.runTime << "s " << (r.verificationResult ? "SUCCESS" : "FAILED ") << endl;

            cout << "#  Download (avg) " << fixed << setprecision(FLOAT_PRECISION) << batch->avgDownloadTime << "s" << endl;
            cout << "#  (Cleanup)      " << fixed << setprecision(FLOAT_PRECISION) << batch->cleanupTime << "s" << endl;
            cout << "#  Fastest        " << fixed << setprecision(FLOAT_PRECISION) << (batch->fastest->uploadTime + batch->fastest->runTime + batch->fastest->downloadTime) << "s " << "(WG size: " << batch->fastest->wgSize << ") " << endl;
            cout << "###############################################################################" << endl;
            cout << endl;
        }

        /**
         * Runs an algorithm on the CPU with the given problem size.
         */
        template <template <typename> class Algorithm>
        CPURun* run(size_t size)
        {
            // create algorithm and run stats, prepare input
            Algorithm<T>* alg = new Algorithm<T>();
            CPURun* run = new CPURun(RunType::CPU, alg->getName(), plugin->getTaskDescription(size));

            data = plugin->genInput(size);
            result = plugin->genResult(size);

            // run algorithm
            timer.start();
            alg->run(data, result, size);
            run->runTime = timer.stop();

            // verfiy result
            run->verificationResult = plugin->verifyResult(alg, data, result, size);

            // cleanup
            plugin->freeInput(data);
            plugin->freeResult(result);

            delete alg;

            stats.push_back(run);

            return run;
        }

        template <template <typename> class Algorithm>
        CLBatch* runCL(Context* context, CommandQueue* queue, size_t size, bool useMultipleWorkGroupSizes)
        {
            // create algorithm and batch stats, prepare input
            Algorithm<T>* alg = new Algorithm<T>();
            CLBatch* batch = new CLBatch(context == cpuContext ? RunType::CL_CPU : RunType::CL_GPU, alg->getName(), plugin->getTaskDescription(size));

            data = plugin->genInput(size);
            result = plugin->genResult(size);

            // run custom initialization
            timer.start();
            alg->init(context);
            batch->initTime = timer.stop();


            //size_t maxWorkGroupSize = min(context->getInfo<size_t>(CL_DEVICE_MAX_WORK_GROUP_SIZE), size);
            size_t maxWorkGroupSize = context->getInfo<size_t>(CL_DEVICE_MAX_WORK_GROUP_SIZE);
            if(!useMultipleWorkGroupSizes)
                batch->runs.push_back(uploadRunDownload(alg, context, queue, maxWorkGroupSize, size));
            else
                for(size_t i = 1; i <= maxWorkGroupSize; i <<= 1)
                    batch->runs.push_back(uploadRunDownload(alg, context, queue, i, size));

            // cleanup
            timer.start();
            alg->cleanup();
            batch->cleanupTime = timer.stop();

            // calculate fastest run
            batch->fastest = min_element(batch->runs.begin(), batch->runs.end(), [](CLRun& a, CLRun& b)
            {
                if(a.exceptionOccured || !a.verificationResult)
                    return false;
                if(b.exceptionOccured || !b.verificationResult)
                    return true;
                return a.uploadTime + a.runTime + a.downloadTime < b.uploadTime + b.runTime + b.downloadTime;
            });

            // calculate averages
            batch->avgUploadTime = 0;
            batch->avgRunTime = 0;
            batch->avgDownloadTime = 0;
            for(auto r : batch->runs)
                if(!r.exceptionOccured)
                {
                    batch->avgUploadTime += r.uploadTime;
                    batch->avgRunTime += r.runTime;
                    batch->avgDownloadTime += r.downloadTime;
                }

            batch->avgUploadTime /= batch->runs.size();
            batch->avgRunTime /= batch->runs.size();
            batch->avgDownloadTime /= batch->runs.size();

            // cleanup
            plugin->freeInput(data);
            plugin->freeResult(result);

            delete alg;

            stats.push_back(batch);

            return batch;
        }

        template <template <typename> class Algorithm>
        inline CLRun uploadRunDownload(Algorithm<T>* alg, Context* context, CommandQueue* queue, size_t workGroupSize, size_t size)
        {
            CLRun run;
            run.wgSize = workGroupSize;

            try
            {
                // upload data
                timer.start();
                alg->upload(context, queue, workGroupSize, data, size);
                queue->finish();
                run.uploadTime = timer.stop();

                // run algorithm
                timer.start();
                alg->run(queue, workGroupSize, size);
                queue->finish();
                run.runTime = timer.stop();

                // download data
                timer.start();
                alg->download(queue, result, size);
                queue->finish();
                run.downloadTime = timer.stop();

                // verify
                run.verificationResult = plugin->verifyResult(alg, data, result, size);
            }
            catch(OpenCLException& e)
            {
                run.exceptionOccured = true;
                run.exceptionMsg = e.what();
            }
            catch(...)
            {
                run.exceptionOccured = true;
                run.exceptionMsg = "unkown";
            }

            return run;
        }

        void prepareTest(size_t size)
        {

        }

        void finishTest()
        {

        }

        Context* gpuContext;
        Context* cpuContext;
        CommandQueue* gpuQueue;
        CommandQueue* cpuQueue;

        Plugin<T>* plugin;

        Timer timer;

        void* data;
        void* result;

        bool noCPU;

        vector<Stats*> stats;
};

#endif // RUNNER_H
