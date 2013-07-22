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
#include "gpu/dixxi/RecursiveScan.h"
#include "gpu/dixxi/LocalWorkEfficientBlockScan.h"
#include "gpu/nvidia/Scan.h"
#include "gpu/thesis/NaiveScan.h"

using namespace std;

int main()
{
    try
    {
        //array<int, 5> sizes = { 2<<10, 2<<11, 2<<12, 2<<13, 2<<14, /*2<<15, 2<<16, 2<<17, 2<<18, 2<<19, 2<<20, 2<<21, 2<<22, 2<<23, 2<<24*/ };
        array<size_t, 1> sizes = { 1024 };

        Runner<int, ScanPlugin> runner(1, sizes.begin(), sizes.end());

        //runner.writeGPUDeviceInfo("gpuinfo.csv");

        runner.start("stats.csv");

        runner.run<cpu::Scan>();

        //runner.run<gpu::clpp::Scan>(CLRunType::GPU); // not working
        //runner.run<gpu::gpugems::LocalNaiveScan>(CLRunType::GPU);
        runner.run<gpu::gpugems::LocalWorkEfficientScan>(CLRunType::GPU);
        //runner.run<gpu::apple::Scan>(CLRunType::GPU);
        //runner.run<gpu::dixxi::ScanTask>(CLRunType::GPU);
        //runner.run<gpu::dixxi::NaiveScan>(CLRunType::GPU);
        //runner.run<gpu::dixxi::WorkEfficientScan>(CLRunType::GPU);
        //runner.run<gpu::dixxi::RecursiveScan>(CLRunType::GPU);
        runner.run<gpu::dixxi::LocalWorkEfficientBlockScan>(CLRunType::GPU);
        //runner.run<gpu::nvidia::Scan>(CLRunType::GPU);
        //runner.run<gpu::thesis::NaiveScan>(CLRunType::GPU);

        runner.finish();
    }
    catch(const exception& e)
    {
        cerr << e.what() << endl;
    }

    getchar();

    return 0;
}
