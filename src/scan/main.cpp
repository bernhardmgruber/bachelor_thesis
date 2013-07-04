#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <array>

#include "../common/Runner.h"
#include "ScanPlugin.h"

#include "cpu/Scan.h"
#include "gpu/clpp/Scan.h"
#include "gpu/gpugems/NaiveScan.h"
#include "gpu/gpugems/WorkEfficientScan.h"
#include "gpu/gpugems/LocalScan.h"
#include "gpu/apple/Scan.h"
#include "gpu/dixxi/ScanTask.h"
#include "gpu/nvidia/Scan.h"

using namespace std;

int main()
{
    try
    {
        array<int, 5> sizes = { 2<<10, 2<<11, 2<<12, 2<<13, 2<<14, /*2<<15, 2<<16, 2<<17, 2<<18, 2<<19, 2<<20, 2<<21, 2<<22, 2<<23, 2<<24*/ };
        Runner<cl_uint, ScanPlugin> runner(3, sizes.begin(), sizes.end());

        runner.writeGPUDeviceInfo("gpuinfo.csv");

        runner.start("stats.csv");

        runner.run<cpu::Scan>();

        //runner.run<gpu::clpp::Scan>(CLRunType::GPU); // not working
        runner.run<gpu::gpugems::NaiveScan>(CLRunType::GPU);
        runner.run<gpu::gpugems::WorkEfficientScan>(CLRunType::GPU);
        runner.run<gpu::gpugems::LocalScan>(CLRunType::GPU);
        runner.run<gpu::apple::Scan>(CLRunType::GPU);
        //runner.run<gpu::dixxi::ScanTask>(CLRunType::GPU);
        runner.run<gpu::nvidia::Scan>(CLRunType::GPU);

        runner.finish();
    }
    catch(const exception& e)
    {
        cerr << e.what() << endl;
        return 1;
    }

    return 0;
}
