#pragma once

#include <iomanip>
#include <map>
#include <algorithm>
#include <vector>
#include <string>
#include <stdexcept>
#include <iterator>
#include <chrono>
#ifdef __GNUG__
#include <initializer_list>
#endif

#include "OpenCL.h"
#include "CPUAlgorithm.h"
#include "CLAlgorithm.h"
#include "Timer.h"
#include "utils.h"
#include "DeviceInfoWriter.h"
#include "StatsWriter.h"
#include "ConsoleWriter.h"

using namespace std;

enum class CLRunType
{
    CPU,
    GPU
};

/**
* Used to run algorithms.
*/
template <typename T, template <typename> class Plugin>
class Runner
{
public:
#ifdef __GNUG__
    /**
    * Constructor
    */
    Runner(size_t iterations, initializer_list<size_t> sizes, bool validate = true)
        : iterations(iterations), sizes(sizes), validate(validate)
    {
        init();
    }
#endif

    /**
    * Constructor
    */
    template <typename I>
    Runner(size_t iterations, const I begin, const I end, bool validate = true)
        : iterations(iterations), validate(validate)
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

        //for(Range* r : ranges)
        //    delete r;
    }

    void start(string statsFile)
    {
        writer.beginFile(statsFile);

        globalTimer.start();
    }

    void finish()
    {
        double seconds = globalTimer.stop();

        writer.endFile(seconds);
        consoleWriter.endOutput(seconds);
    }

    /**
    * Runs the given algorithm once for every provided problem size.
    * The results of the runs are printed to stdout.
    */
    template <template <typename> class CPUAlgorithm>
    void run()
    {
        CPUAlgorithm<T>* alg = new CPUAlgorithm<T>();

        writer.beginAlgorithm(alg->getName(), RunType::CPU);
        consoleWriter.beginAlgorithm(alg->getName(), RunType::CPU);

        for(size_t size : sizes)
            runCPU(alg, size);

        delete alg;

        writer.endAlgorithm();
        consoleWriter.endAlgorithm();
    }

    /**
    * Runs the given algorithm once for every provided problem size.
    * The results of the runs are printed to stdout.
    */
    template <template <typename> class Algorithm>
    void run(CLRunType runType, bool useAllSupportedWorkGroupSizes = false)
    {
        checkRunTypeAvailable(runType);

        Context* context;
        CommandQueue* queue;
        switch(runType)
        {
        case CLRunType::CPU:
            context = cpuContext;
            queue = cpuQueue;
            break;
        case CLRunType::GPU:
            context = gpuContext;
            queue = gpuQueue;
            break;
        }

        Algorithm<T>* alg = new Algorithm<T>();
        alg->setContext(context);
        alg->setCommandQueue(queue);

        // run custom initialization
        timer.start();
        alg->init();
        double initTime = timer.stop();

        writer.beginAlgorithm(alg->getName(), runType == CLRunType::CPU ? RunType::CL_CPU : RunType::CL_GPU, initTime);
        consoleWriter.beginAlgorithm(alg->getName(), runType == CLRunType::CPU ? RunType::CL_CPU : RunType::CL_GPU, initTime);

        // run algorithm for different problem sizes
        for(size_t size : sizes)
            runCL(alg, context, queue, useAllSupportedWorkGroupSizes, size);

        // cleanup
        timer.start();
        alg->cleanup();
        double cleanupTime = timer.stop();

        delete alg;

        writer.endAlgorithm(cleanupTime);
        consoleWriter.endAlgorithm(cleanupTime);
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

        consoleWriter.beginOutput(iterations, sizes, getTypeName<T>());
    }

    /**
    * Runs an algorithm on the CPU with the given problem size.
    */
    void runCPU(CPUAlgorithm<T>* alg, size_t size)
    {
        // create run stats and prepare input
        CPURun run(plugin->getTaskDescription(size), size);

        run.verificationResult = true;

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
            run.verificationResult = run.verificationResult && (validate ? plugin->verifyResult(dynamic_cast<typename Plugin<T>::AlgorithmType*>(alg), data, result, size) : true);

            // cleanup
            plugin->freeInput(data);
            plugin->freeResult(result);

            run.iterations.push_back(iteration);
        }

        // compute mean
        double sum = 0;
        for(CPUIteration& i : run.iterations)
            sum += i.runTime;
        run.runTimeMean = sum / (double)iterations;

        // compute standard deviation
        sum = 0;
        for(CPUIteration& i : run.iterations)
        {
            double diff = i.runTime - run.runTimeMean;
            sum += diff * diff;
        }
        run.runTimeDeviation = sqrt(sum / (double) iterations);

        writer.writeRun(run);
        consoleWriter.writeRun(run);
    }

    /**
    * Runs an algorithm using OpenCL with the given problem size.
    */
    void runCL(CLAlgorithm<T>* alg, Context* context, CommandQueue* queue, bool useAllSupportedWorkGroupSizes, size_t size)
    {
        // create algorithm and batch stats, prepare input
        CLRun run(plugin->getTaskDescription(size), size);

        data = plugin->genInput(size);
        result = plugin->genResult(size);


        if(useAllSupportedWorkGroupSizes)
            for(size_t i : alg->getSupportedWorkGroupSizes())
                run.runsWithWGSize.push_back(uploadRunDownload(alg, context, queue, i, size));
        else
            run.runsWithWGSize.push_back(uploadRunDownload(alg, context, queue, alg->getOptimalWorkGroupSize(), size));

        // calculate fastest run
        run.fastest = min_element(run.runsWithWGSize.begin(), run.runsWithWGSize.end(), [](CLRunWithWGSize& a, CLRunWithWGSize& b) -> double
        {
            if(a.exceptionOccured || !a.verificationResult)
                return false;
            if(b.exceptionOccured || !b.verificationResult)
                return true;
            return a.uploadTimeMean + a.runTimeMean + a.downloadTimeMean < b.uploadTimeMean + b.runTimeMean + b.downloadTimeMean;
        });

        // calculate averages
        run.avgUploadTime = 0;
        run.avgRunTime = 0;
        run.avgDownloadTime = 0;
        for(auto r : run.runsWithWGSize) {
            if(!r.exceptionOccured)
            {
                run.avgUploadTime += r.uploadTimeMean;
                run.avgRunTime += r.runTimeMean;
                run.avgDownloadTime += r.downloadTimeMean;
            }
        }

        run.avgUploadTime /= run.runsWithWGSize.size();
        run.avgRunTime /= run.runsWithWGSize.size();
        run.avgDownloadTime /= run.runsWithWGSize.size();

        // cleanup
        plugin->freeInput(data);
        plugin->freeResult(result);

        writer.writeRun(run);
        consoleWriter.writeRun(run);
    }

    inline CLRunWithWGSize uploadRunDownload(CLAlgorithm<T>* alg, Context* context, CommandQueue* queue, size_t workGroupSize, size_t size)
    {
        CLRunWithWGSize run;
        run.wgSize = workGroupSize;

        try
        {
            run.verificationResult = true;

            for(size_t i = 0; i < iterations; i++)
            {
                CLIteration iteration;

                // upload data
                timer.start();
                alg->upload(workGroupSize, data, size);
                queue->finish();
                iteration.uploadTime = timer.stop();

                // run algorithm
                timer.start();
                alg->run(workGroupSize, size);
                queue->finish();
                iteration.runTime = timer.stop();

                // download data
                timer.start();
                alg->download(result, size);
                queue->finish();
                iteration.downloadTime = timer.stop();

                // verify
                run.verificationResult = run.verificationResult && (validate ? plugin->verifyResult(dynamic_cast<typename Plugin<T>::AlgorithmType*>(alg), data, result, size) : true);

                run.iterations.push_back(iteration);
            }
        }
        catch(const OpenCLException& e)
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

    /**
    * Checks if the context necessary to run an algorithm is available.
    */
    void checkRunTypeAvailable(CLRunType runType)
    {
        if(runType == CLRunType::GPU && !hasCLGPU())
            throw OpenCLException("No GPU context initialized!");
        if(runType == CLRunType::CPU && !hasCLCPU())
            throw OpenCLException("No CPU context initialized!");
    }

    void writeDeviceInfo(Context* context, string fileName, char sep)
    {
        DeviceInfoWriter::write(context, fileName, sep);
    }

    StatsWriter writer;
    ConsoleWriter consoleWriter;

    Context* gpuContext;
    Context* cpuContext;
    CommandQueue* gpuQueue;
    CommandQueue* cpuQueue;

    Plugin<T>* plugin;

    Timer timer;
    Timer globalTimer;

    T* data;
    T* result;

    //vector<Range*> ranges;

    size_t iterations;
    vector<size_t> sizes;

    bool validate;
};
