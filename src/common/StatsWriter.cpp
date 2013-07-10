#include "utils.h"

#include "StatsWriter.h"

void StatsWriter::beginFile(string fileName, char separator)
{
    sep = separator;

    file = ofstream(fileName);
    file.setf(ios::fixed);

    file << "Runner built on " << __DATE__ << " " << __TIME__ << endl;
    file << "Bernhard Manfred Gruber" << endl;
    //file << "Duration: " << timeToString(seconds) << endl;
    file << endl;
    file << endl;

    file.flush();
}

void StatsWriter::endFile(double seconds)
{
    file << endl;
    file << "Total runtime " << timeToString(seconds);
    file << endl;

    file.close();
}

void StatsWriter::beginAlgorithm(string algorithmName, RunType runType, double initTime)
{
    file << algorithmName << sep << runTypeToString(runType) << endl;

    if(initTime != -1.0)
        file << "init time" << sep << initTime << endl;

    switch(runType)
    {
    case RunType::CPU:
        file << "size" << sep;
        file << "run time mean" << sep;
        file << "run time deviation" << sep;
        file << "result" << endl;
        break;
    case RunType::CL_CPU:
    case RunType::CL_GPU:
        file << "size" << sep;
        //file << "init time" << sep;
        file << "upload time mean " << sep;
        file << "upload time deviation" << sep;
        file << "run time mean" << sep;
        file << "run time deviation" << sep;
        file << "download time mean" << sep;
        file << "download time deviation" << sep;
        //file << "cleanup time" << sep;
        file << "wg size" << sep;
        file << "up run down sum" << sep;
        file << "result" << endl;
        break;
    }

    file.flush();
}

void StatsWriter::endAlgorithm(double cleanupTime)
{
    if(cleanupTime != -1.0)
        file << "cleanup time" << sep << cleanupTime << endl;

    file << endl;

    file.flush();
}

void StatsWriter::writeRun(const CPURun& run)
{
    file << run.size << sep;
    file << run.runTimeMean << sep;
    file << run.runTimeDeviation << sep;
    file << (run.exceptionOccured ? "EXCEPTION" : (run.verificationResult ? "SUCCESS" : "FAILED")) << endl;

    file.flush();
}

void StatsWriter::writeRun(const CLRun& run)
{
    file << run.size << sep;
    //file << run.initTime << sep;
    file << run.fastest->uploadTimeMean << sep;
    file << run.fastest->uploadTimeDeviation << sep;
    file << run.fastest->runTimeMean << sep;
    file << run.fastest->runTimeDeviation << sep;
    file << run.fastest->downloadTimeMean << sep;
    file << run.fastest->downloadTimeDeviation << sep;
    //file << run.cleanupTime << sep;
    file << run.fastest->wgSize << sep;
    file << (run.fastest->uploadTimeMean + run.fastest->runTimeMean + run.fastest->downloadTimeMean) << sep;
    file << (run.fastest->exceptionOccured ? "EXCEPTION" : (run.fastest->verificationResult ? "SUCCESS" : "FAILED")) << endl;

    file.flush();        
}
