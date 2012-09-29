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
#include <iterator>

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
            cout << "Writing stats file to " << fileName << " ... ";

            const char sep = ';';

            ofstream os(fileName);
            os.setf(ios::fixed);

            os << "Runner" << endl;
            os << "Bernhard Manfred Gruber" << endl;
            os << __DATE__ << " " << __TIME__ << endl;
            os << endl;
            os << endl;

            // print results of all previous run tests
            for(Range* r : ranges)
            {
                os << r->algorithmName << sep << runTypeToString(r->runType) << endl;

                if(r->runType == RunType::CPU)
                {
                    os << "size" << sep << "run time" << sep << "result" << endl;

                    for(Stats* s : r->stats)
                    {
                        CPURun* run = static_cast<CPURun*>(s);
                        os << run->size << sep << run->runTime << sep << (run->verificationResult ? "SUCCESS" : "FAILED") << endl;
                    }
                }
                else
                {
                    os << "size" << sep << "init time" << sep << "upload time" << sep << "run time" << sep << "download time " << sep << "cleanup time" << sep << "up run down sum" << sep << "result" << endl;

                    for(Stats* s : r->stats)
                    {
                        CLBatch* batch = static_cast<CLBatch*>(s);
                        os << batch->size << sep << batch->initTime << sep << batch->fastest->uploadTime << sep << batch->fastest->runTime << sep << batch->fastest->downloadTime << sep << batch->cleanupTime << sep << (batch->fastest->uploadTime + batch->fastest->runTime + batch->fastest->downloadTime) << sep << (batch->fastest->verificationResult ? "SUCCESS" : "FAILED") << endl;
                    }
                }

                os << endl;
            }

            os.close();

            cout << "DONE" << endl;
        }

        void writeCPUDeviceInfo(string fileName)
        {
            if(hasCLCPU())
                writeDeviceInfo(cpuContext, fileName, ';');
            else
                throw OpenCLException("No CPU context initialized!");
        }

        void writeGPUDeviceInfo(string fileName)
        {
            writeDeviceInfo(gpuContext, fileName, ';');
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
         * The resulting Range object is stored internally.
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
            ranges.push_back(r);

            delete alg;

            return r;
        }

        /**
         * Runs the given algorithm once for every provided problem size.
         * The resulting Range object is stored internally.
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
            ranges.push_back(r);

            delete alg;

            return r;
        }

        CPURun* run(GPUAlgorithm<T>* alg, size_t size)
        {
            throw runtime_error(string(__FUNCTION__) + " should never be called!");
        }

        /**
         * Runs an algorithm on the CPU with the given problem size.
         */
        CPURun* run(CPUAlgorithm<T>* alg, size_t size)
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
            run->verificationResult = plugin->verifyResult(dynamic_cast<typename Plugin<T>::AlgorithmType*>(alg), data, result, size);

            // cleanup
            plugin->freeInput(data);
            plugin->freeResult(result);

            return run;
        }

        CLBatch* runCL(CPUAlgorithm<T>* alg, Context* context, CommandQueue* queue, bool useMultipleWorkGroupSizes, size_t size)
        {
            throw runtime_error(string(__FUNCTION__) + " should never be called!");
        }

        /**
         * Runs an algorithm using OpenCL with the given problem size.
         */
        CLBatch* runCL(GPUAlgorithm<T>* alg, Context* context, CommandQueue* queue, bool useMultipleWorkGroupSizes, size_t size)
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

        inline CLRun uploadRunDownload(GPUAlgorithm<T>* alg, Context* context, CommandQueue* queue, size_t workGroupSize, size_t size)
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
                run.verificationResult = plugin->verifyResult(dynamic_cast<typename Plugin<T>::AlgorithmType*>(alg), data, result, size);
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

        void writeDeviceInfo(Context* context, string fileName, char sep)
        {
            cout << "Writing device info file to " << fileName << " ... ";

            ofstream os(fileName);

            os << "CL_DEVICE_ADDRESS_BITS" << sep << context->getInfo<cl_uint>(CL_DEVICE_ADDRESS_BITS) << endl;
            os << "CL_DEVICE_AVAILABLE" << sep << context->getInfo<cl_bool>(CL_DEVICE_AVAILABLE) << endl;
            os << "CL_DEVICE_COMPILER_AVAILABLE" << sep << context->getInfo<cl_bool>(CL_DEVICE_AVAILABLE) << endl;

            {
                os << "CL_DEVICE_DOUBLE_FP_CONFIG" << sep;
                cl_device_fp_config flags = context->getInfo<cl_device_fp_config>(CL_DEVICE_DOUBLE_FP_CONFIG);
                if(flags & CL_FP_DENORM)
                    os << "CL_FP_DENORM ";
                if(flags & CL_FP_INF_NAN)
                    os << "CL_FP_INF_NAN ";
                if(flags & CL_FP_ROUND_TO_NEAREST)
                    os << "CL_FP_ROUND_TO_NEAREST ";
                if(flags & CL_FP_ROUND_TO_ZERO)
                    os << "CL_FP_ROUND_TO_ZERO ";
                if(flags & CL_FP_ROUND_TO_INF)
                    os << "CL_FP_ROUND_TO_INF ";
                if(flags & CL_FP_FMA)
                    os << "CL_FP_FMA ";
                os << endl;
            }

            os << "CL_DEVICE_ENDIAN_LITTLE" << sep << context->getInfo<cl_bool>(CL_DEVICE_ENDIAN_LITTLE) << endl;
            os << "CL_DEVICE_ERROR_CORRECTION_SUPPORT" << sep << context->getInfo<cl_bool>(CL_DEVICE_ERROR_CORRECTION_SUPPORT) << endl;

            {
                os << "CL_DEVICE_EXECUTION_CAPABILITIES" << sep;
                cl_device_exec_capabilities flags = context->getInfo<cl_device_exec_capabilities>(CL_DEVICE_EXECUTION_CAPABILITIES);
                if(flags & CL_EXEC_KERNEL)
                    os << "CL_EXEC_KERNEL ";
                if(flags & CL_EXEC_NATIVE_KERNEL)
                    os << "CL_EXEC_NATIVE_KERNEL ";
                os << endl;
            }

            os << "CL_DEVICE_EXTENSIONS" << sep << context->getInfo<string>(CL_DEVICE_EXTENSIONS) << endl;
            os << "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE" << sep << context->getInfo<cl_ulong>(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE) << endl;

            {
                os << "CL_DEVICE_GLOBAL_MEM_CACHE_TYPE" << sep;
                cl_device_mem_cache_type cacheType = context->getInfo<cl_device_mem_cache_type>(CL_DEVICE_GLOBAL_MEM_CACHE_TYPE);
                switch(cacheType)
                {
                    case CL_NONE:
                        os << "CL_NONE";
                        break;
                    case CL_READ_ONLY_CACHE:
                        os << "CL_READ_ONLY_CACHE";
                        break;
                    case CL_READ_WRITE_CACHE:
                        os << "CL_READ_WRITE_CACHE";
                        break;
                }
                os << endl;
            }

            os << "CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE" << sep << context->getInfo<cl_uint>(CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE) << endl;
            os << "CL_DEVICE_GLOBAL_MEM_SIZE" << sep << context->getInfo<cl_ulong>(CL_DEVICE_GLOBAL_MEM_SIZE) << endl;

            {
                os << "CL_DEVICE_HALF_FP_CONFIG" << sep;
                cl_device_fp_config flags = context->getInfo<cl_device_fp_config>(CL_DEVICE_HALF_FP_CONFIG);
                if(flags & CL_FP_DENORM)
                    os << "CL_FP_DENORM ";
                if(flags & CL_FP_INF_NAN)
                    os << "CL_FP_INF_NAN ";
                if(flags & CL_FP_ROUND_TO_NEAREST)
                    os << "CL_FP_ROUND_TO_NEAREST ";
                if(flags & CL_FP_ROUND_TO_ZERO)
                    os << "CL_FP_ROUND_TO_ZERO ";
                if(flags & CL_FP_ROUND_TO_INF)
                    os << "CL_FP_ROUND_TO_INF ";
                if(flags & CL_FP_FMA)
                    os << "CL_FP_FMA ";
                if(flags & CL_FP_SOFT_FLOAT)
                    os << "CL_FP_SOFT_FLOAT ";
                os << endl;
            }

            os << "CL_DEVICE_HOST_UNIFIED_MEMORY" << sep << context->getInfo<cl_bool>(CL_DEVICE_HOST_UNIFIED_MEMORY) << endl;
            os << "CL_DEVICE_IMAGE_SUPPORT" << sep << context->getInfo<cl_bool>(CL_DEVICE_IMAGE_SUPPORT) << endl;
            os << "CL_DEVICE_IMAGE2D_MAX_HEIGHT" << sep << context->getInfo<size_t>(CL_DEVICE_IMAGE2D_MAX_HEIGHT) << endl;
            os << "CL_DEVICE_IMAGE2D_MAX_WIDTH" << sep << context->getInfo<size_t>(CL_DEVICE_IMAGE2D_MAX_WIDTH) << endl;
            os << "CL_DEVICE_IMAGE3D_MAX_DEPTH" << sep << context->getInfo<size_t>(CL_DEVICE_IMAGE3D_MAX_DEPTH) << endl;
            os << "CL_DEVICE_IMAGE3D_MAX_HEIGHT" << sep << context->getInfo<size_t>(CL_DEVICE_IMAGE3D_MAX_HEIGHT) << endl;
            os << "CL_DEVICE_IMAGE3D_MAX_WIDTH" << sep << context->getInfo<size_t>(CL_DEVICE_IMAGE3D_MAX_WIDTH) << endl;
            os << "CL_DEVICE_LOCAL_MEM_SIZE" << sep << context->getInfo<cl_ulong>(CL_DEVICE_LOCAL_MEM_SIZE) << endl;

            {
                os << "CL_DEVICE_LOCAL_MEM_TYPE" << sep;
                cl_device_local_mem_type memType = context->getInfo<cl_device_local_mem_type>(CL_DEVICE_LOCAL_MEM_TYPE);
                switch(memType)
                {
                    case CL_LOCAL:
                        os << "CL_LOCAL";
                        break;
                    case CL_GLOBAL:
                        os << "CL_GLOBAL";
                        break;
                }
                os << endl;
            }

            os << "CL_DEVICE_MAX_CLOCK_FREQUENCY" << sep << context->getInfo<cl_uint>(CL_DEVICE_MAX_CLOCK_FREQUENCY) << endl;
            os << "CL_DEVICE_MAX_COMPUTE_UNITS" << sep << context->getInfo<cl_uint>(CL_DEVICE_MAX_COMPUTE_UNITS) << endl;
            os << "CL_DEVICE_MAX_CONSTANT_ARGS" << sep << context->getInfo<cl_uint>(CL_DEVICE_MAX_CONSTANT_ARGS) << endl;
            os << "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE" << sep << context->getInfo<cl_ulong>(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE) << endl;
            os << "CL_DEVICE_MAX_MEM_ALLOC_SIZE" << sep << context->getInfo<cl_ulong>(CL_DEVICE_MAX_MEM_ALLOC_SIZE) << endl;
            os << "CL_DEVICE_MAX_PARAMETER_SIZE" << sep << context->getInfo<size_t>(CL_DEVICE_MAX_PARAMETER_SIZE) << endl;
            os << "CL_DEVICE_MAX_READ_IMAGE_ARGS" << sep << context->getInfo<cl_uint>(CL_DEVICE_MAX_READ_IMAGE_ARGS) << endl;
            os << "CL_DEVICE_MAX_SAMPLERS" << sep << context->getInfo<cl_uint>(CL_DEVICE_MAX_SAMPLERS) << endl;
            os << "CL_DEVICE_MAX_WORK_GROUP_SIZE" << sep << context->getInfo<size_t>(CL_DEVICE_MAX_WORK_GROUP_SIZE) << endl;

            {
                cl_uint dimensions = context->getInfo<cl_uint>(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS);
                os << "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS" << sep << dimensions << endl;

                os << "CL_DEVICE_MAX_WORK_ITEM_SIZES" << sep;
                size_t* sizes = (size_t*)context->getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES);
                copy(sizes, sizes + dimensions, ostream_iterator<size_t>(os, ","));
                delete sizes;
                os << endl;
            }

            os << "CL_DEVICE_MAX_WRITE_IMAGE_ARGS" << sep << context->getInfo<cl_uint>(CL_DEVICE_MAX_WRITE_IMAGE_ARGS) << endl;
            os << "CL_DEVICE_MEM_BASE_ADDR_ALIGN" << sep << context->getInfo<cl_uint>(CL_DEVICE_MEM_BASE_ADDR_ALIGN) << endl;
            os << "CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE" << sep << context->getInfo<cl_uint>(CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE) << endl;
            os << "CL_DEVICE_NAME" << sep << context->getInfo<string>(CL_DEVICE_NAME) << endl;
            os << "CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR" << sep << context->getInfo<cl_uint>(CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR) << endl;
            os << "CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT" << sep << context->getInfo<cl_uint>(CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT) << endl;
            os << "CL_DEVICE_NATIVE_VECTOR_WIDTH_INT" << sep << context->getInfo<cl_uint>(CL_DEVICE_NATIVE_VECTOR_WIDTH_INT) << endl;
            os << "CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG" << sep << context->getInfo<cl_uint>(CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG) << endl;
            os << "CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT" << sep << context->getInfo<cl_uint>(CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT) << endl;
            os << "CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE" << sep << context->getInfo<cl_uint>(CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE) << endl;
            os << "CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF" << sep << context->getInfo<cl_uint>(CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF) << endl;
            os << "CL_DEVICE_OPENCL_C_VERSION" << sep << context->getInfo<string>(CL_DEVICE_OPENCL_C_VERSION) << endl;
            os << "CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR" << sep << context->getInfo<cl_uint>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR) << endl;
            os << "CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT" << sep << context->getInfo<cl_uint>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT) << endl;
            os << "CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT" << sep << context->getInfo<cl_uint>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT) << endl;
            os << "CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG" << sep << context->getInfo<cl_uint>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG) << endl;
            os << "CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT" << sep << context->getInfo<cl_uint>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT) << endl;
            os << "CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE" << sep << context->getInfo<cl_uint>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE) << endl;
            os << "CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF" << sep << context->getInfo<cl_uint>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF) << endl;
            os << "CL_DEVICE_PROFILE" << sep << context->getInfo<string>(CL_DEVICE_PROFILE) << endl;
            os << "CL_DEVICE_PROFILING_TIMER_RESOLUTION" << sep << context->getInfo<size_t>(CL_DEVICE_PROFILING_TIMER_RESOLUTION) << endl;

            {
                os << "CL_DEVICE_QUEUE_PROPERTIES" << sep;
                cl_command_queue_properties properties = context->getInfo<cl_command_queue_properties>(CL_DEVICE_QUEUE_PROPERTIES);
                if(properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
                    os << "CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE ";
                if(properties & CL_QUEUE_PROFILING_ENABLE)
                    os << "CL_QUEUE_PROFILING_ENABLE ";
                os << endl;
            }

            {
                os << "CL_DEVICE_SINGLE_FP_CONFIG" << sep;
                cl_device_fp_config flags = context->getInfo<cl_device_fp_config>(CL_DEVICE_SINGLE_FP_CONFIG);
                if(flags & CL_FP_DENORM)
                    os << "CL_FP_DENORM ";
                if(flags & CL_FP_INF_NAN)
                    os << "CL_FP_INF_NAN ";
                if(flags & CL_FP_ROUND_TO_NEAREST)
                    os << "CL_FP_ROUND_TO_NEAREST ";
                if(flags & CL_FP_ROUND_TO_ZERO)
                    os << "CL_FP_ROUND_TO_ZERO ";
                if(flags & CL_FP_ROUND_TO_INF)
                    os << "CL_FP_ROUND_TO_INF ";
                if(flags & CL_FP_FMA)
                    os << "CP_FP_FMA ";
                if(flags & CL_FP_SOFT_FLOAT)
                    os << "CL_FP_SOFT_FLOAT ";
                os << endl;
            }

            {
                os << "CL_DEVICE_TYPE" << sep;
                cl_device_type deviceType = context->getInfo<cl_device_type>(CL_DEVICE_TYPE);
                switch(deviceType)
                {
                    case CL_DEVICE_TYPE_CPU:
                        os << "CL_DEVICE_TYPE_CPU";
                        break;
                    case CL_DEVICE_TYPE_GPU:
                        os << "CL_DEVICE_TYPE_GPU";
                        break;
                    case CL_DEVICE_TYPE_ACCELERATOR:
                        os << "CL_DEVICE_TYPE_ACCELERATOR";
                        break;
                    case CL_DEVICE_TYPE_DEFAULT:
                        os << "CL_DEVICE_TYPE_DEFAULT";
                        break;
                }
                os << endl;
            }

            os << "CL_DEVICE_VENDOR" << sep << context->getInfo<string>(CL_DEVICE_VENDOR) << endl;
            os << "CL_DEVICE_VENDOR_ID" << sep << context->getInfo<cl_uint>(CL_DEVICE_VENDOR_ID) << endl;
            os << "CL_DEVICE_VERSION" << sep << context->getInfo<string>(CL_DEVICE_VERSION) << endl;
            os << "CL_DRIVER_VERSION" << sep << context->getInfo<string>(CL_DRIVER_VERSION) << endl;

            os.close();

            cout << "DONE" << endl;
        }

        Context* gpuContext;
        Context* cpuContext;
        CommandQueue* gpuQueue;
        CommandQueue* cpuQueue;

        Plugin<T>* plugin;

        Timer timer;

        T* data;
        T* result;

        bool noCPU;

        vector<Range*> ranges;
};

#endif // RUNNER_H
