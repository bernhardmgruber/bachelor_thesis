#include <iostream>
#include <iomanip>

#include "utils.h"

#include "ConsoleWriter.h"

using namespace std;

const int ConsoleWriter::FLOAT_PRECISION = 3;

void ConsoleWriter::beginOutput(size_t iterations, vector<size_t> sizes, string typeName)
{
    cout << "##### Initialized Runner #####" << endl;
    cout << "Running " << iterations << " iterations " << endl;
    cout << "Sizes: ";
    copy(sizes.begin(), sizes.end(), ostream_iterator<size_t>(cout, ", "));
    cout << "\b\b " << endl;
    cout << "Type: " << typeName << endl;
    cout << endl;
}

void ConsoleWriter::endOutput(double seconds)
{
    cout << "Finishing runner after " << seconds << "s of run time" << endl;
}

void ConsoleWriter::beginAlgorithm(string algorithmName, RunType runType)
{
    cout << "###############################################################################" << endl;
    cout << "# " << algorithmName << " " << runTypeToString(runType) << endl;
}

void ConsoleWriter::endAlgorithm()
{
    cout << "###############################################################################" << endl;
    cout << endl;
}

void ConsoleWriter::writeRun(const CPURun& run)
{
    cout << "#  " << run.taskDescription << endl;
    if(run.exceptionOccured)
        cout << "#  Run            " << "EXCEPTION: " << run.exceptionMsg << endl;
    else
        cout << "#  Run            " << fixed << setprecision(FLOAT_PRECISION) << run.runTimeMean << "s (sigma " << run.runTimeDeviation << "s) " << (run.verificationResult ? "SUCCESS" : "FAILED") << endl;
}

void ConsoleWriter::writeRun(const CLRun& run)
{
    cout << "#  " << run.taskDescription << endl;
    cout << "#  (Init)         " << fixed << setprecision(FLOAT_PRECISION) << run.initTime << "s" << endl;
    cout << "#  Upload (avg)   " << fixed << setprecision(FLOAT_PRECISION) << run.avgUploadTime << "s" << endl;

    for(auto r : run.runsWithWGSize)
        if(r.exceptionOccured)
            cout << "#  WG: " << setw(4) << r.wgSize << "       EXCEPTION: " << r.exceptionMsg << endl;
        else
            cout << "#  WG: " << setw(4) << r.wgSize << "       " << fixed << setprecision(FLOAT_PRECISION) << r.runTimeMean << "s (sigma " << r.runTimeDeviation << "s) " << (r.verificationResult ? "SUCCESS" : "FAILED ") << endl;

    cout << "#  Download (avg) " << fixed << setprecision(FLOAT_PRECISION) << run.avgDownloadTime << "s" << endl;
    cout << "#  (Cleanup)      " << fixed << setprecision(FLOAT_PRECISION) << run.cleanupTime << "s" << endl;
    cout << "#  Fastest        " << fixed << setprecision(FLOAT_PRECISION) << (run.fastest->uploadTimeMean + run.fastest->runTimeMean + run.fastest->downloadTimeMean) << "s " << "(WG: " << run.fastest->wgSize << ") " << endl;
}
