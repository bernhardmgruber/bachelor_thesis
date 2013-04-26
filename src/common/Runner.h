#ifndef RUNNER_H
#define RUNNER_H

#include <typeinfo>
#include <iomanip>
#include <map>
#include <algorithm>
#include <vector>
#include <string>
#include <stdexcept>
#include <iterator>
#ifdef __GNUG__
#include <initializer_list>
#endif

#include "OpenCL.h"
#include "CPUAlgorithm.h"
#include "GPUAlgorithm.h"
#include "Timer.h"
#include "utils.h"
#include "StatsWriter.h"

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
    static const int FLOAT_PRECISION = 3;

#ifdef __GNUG__
    /**
    * Constructor
    */
    Runner(size_t iterations, initializer_list<size_t> sizes, bool validate = true)
        : iterations(iterations), sizes(sizes)
    {
        init();
    }
#endif

    template <typename I>
    Runner(size_t iterations, const I begin, const I end, bool validate = true)
        : iterations(iterations)
    {
        copy(begin, end, back_inserter(sizes));
        init();
    }

    /**
    * Destructor
    */
    virtual ~Runner()
    {
        if(hasCLGPU())
        {
            delete gpuContext;
            delete gpuQueue;
        }
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
    * Runs the given algorithm once for every provided problem size.
    * The results of the runs are printed to stdout.
    */
    template <template <typename> class Algorithm>
    void run(RunType runType, bool useMultipleWorkGroupSizes = true)
    {
        checkRunTypeAvailable(runType);

        Algorithm<T>* alg = new Algorithm<T>();
        Range* r = new Range(runType, alg->getName());

        for(size_t size : sizes)
        {
            Stats* s;
            switch(runType)
            {
            case CPU:
                s = runCPU(alg, size);
                break;
            case CL_CPU:
                if(!hasCLCPU())
                    throw OpenCLException("No CPU context initialized!");
                s = runCL(alg, cpuContext, cpuQueue, useMultipleWorkGroupSizes, size);
                break;
            case CL_GPU:
                s = runCL(alg, gpuContext, gpuQueue, useMultipleWorkGroupSizes, size);
                break;
            }
            r->stats.push_back(s);
        }
        ranges.push_back(r);

        delete alg;

        printRange(r, runType);
    }

    /**
    * Writes the results of all previous runs to the given file.
    */
    void writeStats(string fileName)
    {
        double seconds = globalTimer.stop();

        cout << "Writing stats file to " << fileName << " ... ";

        const char sep = ';';

        ofstream os(fileName);
        os.setf(ios::fixed);

        os << "Runner " << __DATE__ << " " << __TIME__ << endl;
        os << "Bernhard Manfred Gruber" << endl;
        os << "Duration: " << timeToString(seconds) << endl;
        os << endl;
        os << endl;

        // print results of all previous run tests
        for(Range* r : ranges)
        {
            os << r->algorithmName << sep << runTypeToString(r->runType) << endl;

            if(r->runType == RunType::CPU)
            {
                os << "size" << sep << "run time mean" << sep << "run time deviation" << sep << "result" << endl;

                for(Stats* s : r->stats)
                {
                    CPURun* run = static_cast<CPURun*>(s);
                    os << run->size << sep << run->runTimeMean << sep << run->runTimeDeviation << sep << (run->exceptionOccured ? "EXCEPTION" : (run->verificationResult ? "SUCCESS" : "FAILED")) << endl;
                }
            }
            else
            {
                os << "size" << sep << "init time" << sep << "upload time mean " << sep << "upload time deviation" << sep << "run time mean" << sep << "run time deviation" << sep << "download time mean" << sep << "download time deviation" << sep << "cleanup time" << sep << "wg size" << sep << "up run down sum" << sep << "result" << endl;

                for(Stats* s : r->stats)
                {
                    CLBatch* batch = static_cast<CLBatch*>(s);
                    os << batch->size << sep << batch->initTime << sep << batch->fastest->uploadTimeMean << sep << batch->fastest->uploadTimeDeviation << sep << batch->fastest->runTimeMean << sep << batch->fastest->runTimeDeviation << sep << batch->fastest->downloadTimeMean << sep << batch->fastest->downloadTimeDeviation << sep << batch->cleanupTime << sep << batch->fastest->wgSize << sep << (batch->fastest->uploadTimeMean + batch->fastest->runTimeMean + batch->fastest->downloadTimeMean) << sep << (batch->fastest->exceptionOccured ? "EXCEPTION" : (batch->fastest->verificationResult ? "SUCCESS" : "FAILED")) << endl;
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
        if(hasCLGPU())
            writeDeviceInfo(gpuContext, fileName, ';');
        else
            throw OpenCLException("No GPU context initialized!");
    }

    bool hasCLCPU()
    {
        return cpuContext != nullptr;
    }

    bool hasCLGPU()
    {
        return gpuContext != nullptr;
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

    struct CPUIteration
    {
        double runTime;
    };

    struct CPURun : public Stats
    {
        vector<CPUIteration> iterations;
        double runTimeMean;
        double runTimeDeviation;
        bool verificationResult;
        bool exceptionOccured;
        string exceptionMsg;

        CPURun(string taskDescription, size_t size)
            : Stats(taskDescription, size), exceptionOccured(false)
        {
        }
    };

    struct CLIteration
    {
        double uploadTime;
        double runTime;
        double downloadTime;
    };

    struct CLRun
    {
        size_t wgSize;
        vector<CLIteration> iterations;
        double uploadTimeMean;
        double uploadTimeDeviation;
        double runTimeMean;
        double runTimeDeviation;
        double downloadTimeMean;
        double downloadTimeDeviation;
        bool verificationResult;
        bool exceptionOccured;
        string exceptionMsg;

        CLRun()
            : exceptionOccured(false)
        {
        }
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

    void init()
    {
        OpenCL::init();

        try
        {
            gpuContext = OpenCL::getGPUContext();
        }
        catch(OpenCLException& e)
        {
            cerr << "Failed to initialize GPU context: " << e.what() << endl;
            gpuContext = nullptr;
        }

        try
        {
            cpuContext = OpenCL::getCPUContext();
        }
        catch(OpenCLException& e)
        {
            cerr << "Failed to initialize CPU context: " << e.what() << endl;
            cpuContext = nullptr;
        }

        if(gpuContext)
            gpuQueue = gpuContext->createCommandQueue();
        if(cpuContext)
            cpuQueue = cpuContext->createCommandQueue();

        plugin = new Plugin<T>();

        cout << "##### Initialized Runner #####" << endl;
        cout << "Running " << iterations << " iterations " << endl;
        cout << "Sizes: ";
        copy(sizes.begin(), sizes.end(), ostream_iterator<size_t>(cout, ", "));
        cout << "\b\b " << endl;
        cout << "Type: " << getTypeName<T>() << endl;
        cout << endl;

        globalTimer.start();
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
                    cout << "#  CPU       " << fixed << setprecision(FLOAT_PRECISION) << run->runTimeMean << " (sigma " << run->runTimeDeviation << "s) " << (run->verificationResult ? "SUCCESS" : "FAILED") << endl;
            }

            cout << "###############################################################################" << endl;
            cout << endl;
        }
        else
        {
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
                        cout << "#  " << (runType == RunType::CL_CPU ? "C" : "G") << "PU (WG: " << setw(4) << r.wgSize << ") EXCEPTION: " << r.exceptionMsg << endl;
                    else
                        cout << "#  " << (runType == RunType::CL_CPU ? "C" : "G") << "PU (WG: " << setw(4) << r.wgSize << ") " << fixed << setprecision(FLOAT_PRECISION) << r.runTimeMean << "s (sigma " << r.runTimeDeviation << "s) " << (r.verificationResult ? "SUCCESS" : "FAILED ") << endl;

                cout << "#  Download (avg) " << fixed << setprecision(FLOAT_PRECISION) << batch->avgDownloadTime << "s" << endl;
                cout << "#  (Cleanup)      " << fixed << setprecision(FLOAT_PRECISION) << batch->cleanupTime << "s" << endl;
                cout << "#  Fastest        " << fixed << setprecision(FLOAT_PRECISION) << (batch->fastest->uploadTimeMean + batch->fastest->runTimeMean + batch->fastest->downloadTimeMean) << "s " << "(WG: " << batch->fastest->wgSize << ") " << endl;
            }

            cout << "###############################################################################" << endl;
            cout << endl;
        }
    }

    CPURun* runCPU(GPUAlgorithm<T>* alg, size_t size)
    {
        throw runtime_error(string(__FUNCTION__) + " should never be called!");
    }

    /**
    * Runs an algorithm on the CPU with the given problem size.
    */
    CPURun* runCPU(CPUAlgorithm<T>* alg, size_t size)
    {
        // create run stats and prepare input
        CPURun* run = new CPURun(plugin->getTaskDescription(size), size);

        run->verificationResult = true;

        for(size_t i = 0; i < iterations; i++)
        {
            CPUIteration iteration;

            data = plugin->genInput(size);
            result = plugin->genResult(size);

            // run algorithm
            timer.start();
            alg->run(data, result, size);
            iteration.runTime = timer.stop();

            // verfiy result
            run->verificationResult = run->verificationResult && (validate ? plugin->verifyResult(dynamic_cast<typename Plugin<T>::AlgorithmType*>(alg), data, result, size) : true);

            // cleanup
            plugin->freeInput(data);
            plugin->freeResult(result);

            run->iterations.push_back(iteration);
        }

        // compute mean
        double sum = 0;
        for(CPUIteration& i : run->iterations)
            sum += i.runTime;
        run->runTimeMean = sum / (double)iterations;

        // compute standard deviation
        sum = 0;
        for(CPUIteration& i : run->iterations)
        {
            double diff = i.runTime - run->runTimeMean;
            sum += diff * diff;
        }
        run->runTimeDeviation = sqrt(sum / (double) iterations);

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
            return a.uploadTimeMean + a.runTimeMean + a.downloadTimeMean < b.uploadTimeMean + b.runTimeMean + b.downloadTimeMean;
        });

        // calculate averages
        batch->avgUploadTime = 0;
        batch->avgRunTime = 0;
        batch->avgDownloadTime = 0;
        for(auto r : batch->runs)
            if(!r.exceptionOccured)
            {
                batch->avgUploadTime += r.uploadTimeMean;
                batch->avgRunTime += r.runTimeMean;
                batch->avgDownloadTime += r.downloadTimeMean;
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
            run.verificationResult = true;

            for(size_t i = 0; i < iterations; i++)
            {
                CLIteration iteration;

                // upload data
                timer.start();
                alg->upload(context, queue, workGroupSize, data, size);
                queue->finish();
                iteration.uploadTime = timer.stop();

                // run algorithm
                timer.start();
                alg->run(queue, workGroupSize, size);
                queue->finish();
                iteration.runTime = timer.stop();

                // download data
                timer.start();
                alg->download(queue, result, size);
                queue->finish();
                iteration.downloadTime = timer.stop();

                // verify
                run.verificationResult = run.verificationResult && (validate ? plugin->verifyResult(dynamic_cast<typename Plugin<T>::AlgorithmType*>(alg), data, result, size) : true);

                run.iterations.push_back(iteration);
            }
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

        // compute means
        double uploadSum = 0;
        double runSum = 0;
        double downloadSum = 0;
        for(CLIteration& i : run.iterations)
        {
            uploadSum += i.uploadTime;
            runSum += i.runTime;
            downloadSum += i.downloadTime;
        }
        run.uploadTimeMean = uploadSum / (double)iterations;
        run.runTimeMean = runSum / (double)iterations;
        run.downloadTimeMean = downloadSum / (double)iterations;

        // compute standard deviation
        uploadSum = 0;
        runSum = 0;
        downloadSum = 0;
        for(CLIteration& i : run.iterations)
        {
            double diff;
            diff = i.uploadTime - run.uploadTimeMean;
            uploadSum += diff * diff;
            diff = i.runTime - run.runTimeMean;
            runSum += diff * diff;
            diff = i.downloadTime - run.downloadTimeMean;
            downloadSum += diff * diff;
        }
        run.uploadTimeDeviation = sqrt(uploadSum / (double) iterations);
        run.runTimeDeviation = sqrt(runSum / (double) iterations);
        run.downloadTimeDeviation = sqrt(downloadSum / (double) iterations);

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

    /**
    * Checks if the context necessary to run an algorithm is available.
    */
    void checkRunTypeAvailable(RunType runType)
    {
        if(runType == CL_GPU && !hasCLGPU())
            throw OpenCLException("No GPU context initialized!");
        if(runType == CL_CPU && !hasCLCPU())
            throw OpenCLException("No CPU context initialized!");
    }

    void writeDeviceInfo(Context* context, string fileName, char sep)
    {
        StatsWriter::Write(context, fileName, sep);
    }

    Context* gpuContext;
    Context* cpuContext;
    CommandQueue* gpuQueue;
    CommandQueue* cpuQueue;

    Plugin<T>* plugin;

    Timer timer;
    Timer globalTimer;

    T* data;
    T* result;

    vector<Range*> ranges;

    size_t iterations;
    vector<size_t> sizes;

    bool validate;
};

#endif // RUNNER_H
