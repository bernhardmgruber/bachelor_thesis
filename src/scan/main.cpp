#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <array>
#include <set>

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

#define MAX_POWER_OF_TWO 26
#define RESOLUTION 5

int main()
{
    try
    {
        //set<size_t> sizes;

        //for(int i = 1 * RESOLUTION; i <= MAX_POWER_OF_TWO * RESOLUTION; i++)
        //{
        //    size_t s = (size_t)pow(2.0, (double)i / (double)RESOLUTION);
        //    sizes.insert(s);
        //}

        //array<int, 26> sizes = { 1<<1, 1<<2, 1<<3, 1<<4, 1<<5, 1<<6, 1<<7,1<<8, 1<<9, 1<<10, 1<<11, 1<<12, 1<<13, 1<<14, 1<<15, 1<<16, 1<<17, 1<<18, 1<<19, 1<<20, 1<<21, 1<<22, 1<<23, 1<<24, 1<<25, 1<<26 };
        array<size_t, 1> sizes = { 1<<26 };

        Runner<cl_int, ScanPlugin> runner(3, sizes.begin(), sizes.end());

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
        //runner.run<gpu::thesis::NaiveScan>(CLRunType::GPU);
        //runner.run<gpu::thesis::WorkEfficientScan>(CLRunType::GPU);
        //runner.run<gpu::thesis::RecursiveScan>(CLRunType::GPU);
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
