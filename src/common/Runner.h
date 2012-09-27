#ifndef RUNNER_H
#define RUNNER_H

#include "OpenCL.h"
#include "CPUAlgorithm.h"
#include "GPUAlgorithm.h"
#include "Timer.h"
#include "utils.h"

#include <typeinfo>
#include <iomanip>
#include <map>
#include <algorithm>
#include <vector>
#include <string>
#include <stdexcept>

using namespace std;

enum RunType
{
    CPU,
    CL_CPU,
    CL_GPU
};

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

        /**
         * Destructor
         */
        virtual ~Runner()
        {
            delete gpuContext;
            delete gpuQueue;
            if(hasCLCPU())
            {
                delete cpuContext;
                delete cpuQueue;
            }
            OpenCL::cleanup();

            delete plugin;

            for(Range* r : ranges)
                delete r;
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

        /**
         * Runs an algorithm once with the given problem size.
         * The results of the run are printed to stdout.
         */
        template <template <typename> class Algorithm>
        void printOnce(RunType runType, size_t size, bool useMultipleWorkGroupSizes = true)
        {
            Range* r = runOnce<Algorithm>(size, runType, useMultipleWorkGroupSizes);

            printRange(r, runType);
        }

        /**
         * Runs the given algorithm once for every provided problem size.
         * The results of the runs are printed to stdout.
         */
        template <template <typename> class Algorithm>
        void printRange(RunType runType, size_t* range, size_t count, bool useMultipleWorkGroupSizes = true)
        {
            Range* r = runRange<Algorithm>(range, count, runType, useMultipleWorkGroupSizes);

            printRange(r, runType);
        }

        /**
         * Writes the results of all previous runs to the given file.
         */
        void writeStats(string fileName)
        {
            const char sep = ';';

            ofstream os(fileName);

            os << "Runner" << endl;
            os << "Bernhard Manfred Gruber" << endl;
            os << __DATE__ << " " << __TIME__ << endl;
            os << endl;
            os << endl;

            //os << "Algorithm name" << sep << "Type" << endl;

            for(Range* r : ranges)
            {
                os << r->algorithmName << sep << runTypeToString(r->runType) << endl;

                if(r->runType == RunType::CPU)
                {
                    os << "run time" << sep << "result" << endl;

                    for(Stats* s : r->stats)
                    {
                        CPURun* run = static_cast<CPURun*>(s);
                        os << run->runTime << sep << (run->verificationResult ? "SUCCESS" : "FAILED") << endl;
                    }
                }
                else
                {
                    os << "init time" << sep << "upload time" << sep << "run time" << sep << "download time " << sep << "cleanup time" << sep << "result" << endl;

                    for(Stats* s : r->stats)
                    {
                        CLBatch* batch = static_cast<CLBatch*>(s);
                        os << batch->initTime << sep << batch->fastest->uploadTime << sep << batch->fastest->runTime << sep << batch->fastest->downloadTime << batch->cleanupTime << sep << (batch->fastest->verificationResult ? "SUCCESS" : "FAILED") << endl;
                    }
                }

                os << endl;
            }

            os.close();
        }

    private:
        struct Stats
        {
            const string taskDescription;
            size_t size;

            Stats(string taskDescription, size_t size)
                : taskDescription(taskDescription), size(size)
            {
            }
        };

        struct CPURun : public Stats
        {
            double runTime;
            bool verificationResult;
            bool exceptionOccured = false;
            string exceptionMsg;

            CPURun(string taskDescription, size_t size)
                : Stats(taskDescription, size)
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

            CLBatch(string taskDescription, size_t size)
                : Stats(taskDescription, size)
            {
            }
        };

        struct Range
        {
            RunType runType;
            const string algorithmName;

            vector<Stats*> stats;

            Range(RunType runType, string algorithmName)
                : runType(runType), algorithmName(algorithmName)
            {
            }

            ~Range()
            {
                for(Stats* s : stats)
                    delete s;
            }
        };

        /**
         * Prints information about a device.
         */
        void printDevice(Context* context)
        {
            cout << "" << context->getInfo<string>(CL_DEVICE_NAME) << endl;
            cout << "   " << context->getInfo<string>(CL_DEVICE_VENDOR) << endl;
            cout << "   " << context->getInfo<size_t>(CL_DEVICE_MAX_COMPUTE_UNITS) << " compute units" << endl;
            cout << "   " << context->getInfo<size_t>(CL_DEVICE_MAX_CLOCK_FREQUENCY) << " MHz frequency" << endl;
            cout << "   " << fixed << setprecision(FLOAT_PRECISION) << sizeToString(context->getInfo<cl_ulong>(CL_DEVICE_GLOBAL_MEM_SIZE)) << " global mem" << endl;
            cout << "   " << fixed << setprecision(FLOAT_PRECISION) << sizeToString(context->getInfo<cl_ulong>(CL_DEVICE_LOCAL_MEM_SIZE)) << " local mem" << endl;
        }

        void printRange(Range* r, RunType runType)
        {
            if(runType == RunType::CPU)
            {
                cout << "###############################################################################" << endl;
                cout << "# " << r->algorithmName << endl;

                for(Stats* s : r->stats)
                {
                    CPURun* run = (CPURun*) s;

                    cout << "#  " << run->taskDescription << endl;
                    if(run->exceptionOccured)
                        cout << "#  CPU       " << "EXCEPTION: " << run->exceptionMsg << endl;
                    else
                        cout << "#  CPU       " << fixed << setprecision(FLOAT_PRECISION) << run->runTime << "s " << (run->verificationResult ? "SUCCESS" : "FAILED") << endl;
                }

                cout << "###############################################################################" << endl;
                cout << endl;
            }
            else
            {
                if(runType == RunType::CL_CPU)
                    if(!hasCLCPU())
                        throw OpenCLException("No CPU context initialized!");

                // print results
                cout << "###############################################################################" << endl;
                cout << "# " << r->algorithmName << endl;

                for(Stats* s : r->stats)
                {
                    CLBatch* batch = (CLBatch*) s;

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
                }

                cout << "###############################################################################" << endl;
                cout << endl;
            }
        }

        /**
         * Runs the given algorithm once for the given problem size.
         */
        template <template <typename> class Algorithm>
        Range* runOnce(size_t size, RunType runType, bool useMultipleWorkGroupSizes)
        {
            Algorithm<T>* alg = new Algorithm<T>();
            Range* r = new Range(runType, alg->getName());

            Stats* s;

            switch(runType)
            {
                case CPU:
                    s = run(alg, size);
                    break;
                case CL_CPU:
                    s = runCL(alg, cpuContext, cpuQueue, useMultipleWorkGroupSizes, size);
                    break;
                case CL_GPU:
                    s = runCL(alg, gpuContext, gpuQueue, useMultipleWorkGroupSizes, size);
                    break;
            }
            r->stats.push_back(s);

            delete alg;

            return r;
        }

        /**
         * Runs the given algorithm once for every provided problem size.
         */
        template <template <typename> class Algorithm>
        Range* runRange(size_t* range, size_t count, RunType runType, bool useMultipleWorkGroupSizes)
        {
            Algorithm<T>* alg = new Algorithm<T>();
            Range* r = new Range(runType, alg->getName());

            for(size_t i = 0; i < count; i++)
            {
                Stats* s;
                switch(runType)
                {
                    case CPU:
                        s = run(alg, range[i]);
                        break;
                    case CL_CPU:
                        s = runCL(alg, cpuContext, cpuQueue, useMultipleWorkGroupSizes, range[i]);
                        break;
                    case CL_GPU:
                        s = runCL(alg, gpuContext, gpuQueue, useMultipleWorkGroupSizes, range[i]);
                        break;
                }
                r->stats.push_back(s);
            }

            delete alg;

            return r;
        }

        CPURun* run(GPUAlgorithm* alg, size_t size)
        {
            throw runtime_error(string(__FUNCTION__) + " should never be called!");
        }

        /**
         * Runs an algorithm on the CPU with the given problem size.
         */
        CPURun* run(CPUAlgorithm* alg, size_t size)
        {
            // create run stats and prepare input
            CPURun* run = new CPURun(plugin->getTaskDescription(size), size);

            data = plugin->genInput(size);
            result = plugin->genResult(size);

            // run algorithm
            timer.start();
            alg->run(data, result, size);
            run->runTime = timer.stop();

            // verfiy result
            run->verificationResult = plugin->verifyResult((typename Plugin<T>::AlgorithmType*)alg, data, result, size);

            // cleanup
            plugin->freeInput(data);
            plugin->freeResult(result);

            return run;
        }

        CLBatch* runCL(CPUAlgorithm* alg, Context* context, CommandQueue* queue, bool useMultipleWorkGroupSizes, size_t size)
        {
            throw runtime_error(string(__FUNCTION__) + " should never be called!");
        }

        /**
         * Runs an algorithm using OpenCL with the given problem size.
         */
        CLBatch* runCL(GPUAlgorithm* alg, Context* context, CommandQueue* queue, bool useMultipleWorkGroupSizes, size_t size)
        {
            // create algorithm and batch stats, prepare input
            CLBatch* batch = new CLBatch(plugin->getTaskDescription(size), size);

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

            return batch;
        }

        inline CLRun uploadRunDownload(GPUAlgorithm* alg, Context* context, CommandQueue* queue, size_t workGroupSize, size_t size)
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
                run.verificationResult = plugin->verifyResult((typename Plugin<T>::AlgorithmType*)alg, data, result, size);
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

        string runTypeToString(RunType runType)
        {
            switch(runType)
            {
                case CPU:
                    return "CPU";
                case CL_CPU:
                    return "CPU CL";
                case CL_GPU:
                    return "GPU CL";
            }

            return "unkown";
        }

        bool hasCLCPU()
        {
            return cpuContext != nullptr;
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

        vector<Range*> ranges;
};

#endif // RUNNER_H
