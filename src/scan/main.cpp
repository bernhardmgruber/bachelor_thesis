#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <array>

#include "../common/Runner.h"
#include "ScanPlugin.h"

#include "cpu/Scan.h"
#include "gpu/clpp/Scan.h"
#include "gpu/gpugems/LocalNaiveScan.h"
#include "gpu/gpugems/LocalWorkEfficientScan.h"
#include "gpu/apple/Scan.h"
#include "gpu/dixxi/ScanTask.h"
#include "gpu/dixxi/NaiveScan.h"
#include "gpu/dixxi/WorkEfficientScan.h"
#include "gpu/dixxi/WorkEfficientScanWI.h"
#include "gpu/dixxi/RecursiveScan.h"
#include "gpu/dixxi/RecursiveVecScan.h"
#include "gpu/dixxi/LocalWorkEfficientVecScan.h"
#include "gpu/nvidia/Scan.h"
#include "gpu/thesis/NaiveScan.h"
#include "gpu/thesis/WorkEfficientScan.h"
#include "gpu/thesis/RecursiveScan.h"
#include "gpu/thesis/RecursiveVecScan.h"

using namespace std;

int main()
{
    try
    {
        array<int, 1> sizes = { 1<<26 };

        Runner<cl_int, ScanPlugin> runner(1, sizes.begin(), sizes.end());

        //runner.writeGPUDeviceInfo("gpuinfo.csv");

        runner.start("stats.csv");

        //runner.run<cpu::Scan>();

        //runner.run<gpu::clpp::Scan>(CLRunType::GPU); // not working
        //runner.run<gpu::gpugems::LocalNaiveScan>(CLRunType::GPU);
        //runner.run<gpu::gpugems::LocalWorkEfficientScan>(CLRunType::GPU);
        //runner.run<gpu::apple::Scan>(CLRunType::GPU);
        //runner.run<gpu::dixxi::ScanTask>(CLRunType::GPU);
        //runner.run<gpu::dixxi::NaiveScan>(CLRunType::GPU);5
        //runner.run<gpu::dixxi::WorkEfficientScan>(CLRunType::GPU);
        //runner.run<gpu::dixxi::WorkEfficientScanWI>(CLRunType::GPU);
        //runner.run<gpu::dixxi::RecursiveScan>(CLRunType::GPU);
        //runner.run<gpu::dixxi::RecursiveVecScan>(CLRunType::GPU);
        //runner.run<gpu::dixxi::LocalWorkEfficientVecScan>(CLRunType::GPU);
        //runner.run<gpu::nvidia::Scan>(CLRunType::GPU);
        runner.run<gpu::thesis::NaiveScan>(CLRunType::GPU);
        runner.run<gpu::thesis::WorkEfficientScan>(CLRunType::GPU); 
        runner.run<gpu::thesis::RecursiveScan>(CLRunType::GPU);
        runner.run<gpu::thesis::RecursiveVecScan>(CLRunType::GPU);

        runner.finish();
    }
    catch(const exception& e)
    {
        cerr << e.what() << endl;
    }

    getchar();

    return 0;
}
