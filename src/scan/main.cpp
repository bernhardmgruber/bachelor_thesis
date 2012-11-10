#include <CL/cl.h>
#include <iostream>
#include <fstream>

#include "../common/Runner.h"
#include "ScanPlugin.h"

#include "cpu/Scan.h"
#include "gpu/clpp/Scan.h"
#include "gpu/gpugems/NaiveScan.h"
#include "gpu/gpugems/WorkEfficientScan.h"
#include "gpu/gpugems/LocalScan.h"
#include "gpu/apple/Scan.h"
#include "gpu/dixxi/ScanTask.h"

using namespace std;

int main()
{
    try
    {
        Runner<int, ScanPlugin> runner(5, {2<<10, 2<<11, 2<<12, 2<<13, 2<<14, 2<<15, 2<<16, 2<<17, 2<<18, 2<<19, 2<<20});

        runner.run<cpu::Scan>(RunType::CPU);

        //runner.printOnce<gpu::clpp::Scan>(RunType::CL_GPU, size, false); // not working
        runner.run<gpu::gpugems::NaiveScan>(RunType::CL_GPU, false);
        runner.run<gpu::gpugems::WorkEfficientScan>(RunType::CL_GPU, false);
        runner.run<gpu::gpugems::LocalScan>(RunType::CL_GPU, false);
        runner.run<gpu::apple::Scan>(RunType::CL_GPU, false);
        runner.run<gpu::dixxi::ScanTask>(RunType::CL_GPU, false);

        // also run them on the CPU
        runner.run<gpu::gpugems::NaiveScan>(RunType::CL_CPU, false);
        runner.run<gpu::gpugems::WorkEfficientScan>(RunType::CL_CPU, false);
        runner.run<gpu::gpugems::LocalScan>(RunType::CL_CPU, false);
        runner.run<gpu::apple::Scan>(RunType::CL_CPU, false);
        runner.run<gpu::dixxi::ScanTask>(RunType::CL_CPU, false);

        runner.writeStats("stats.csv");
        runner.writeGPUDeviceInfo("gpuinfo.csv");
    }
    catch(OpenCLException& e)
    {
        cerr << e.what() << endl;
        return 1;
    }

    return 0;
}
