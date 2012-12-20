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
#include "gpu/nvidia/Scan.h"

using namespace std;

int main()
{
    try
    {
        Runner<cl_uint, ScanPlugin> runner(5, {2<<10, 2<<11, 2<<12, 2<<13, 2<<14, 2<<15, 2<<16, 2<<17, 2<<18, 2<<19, 2<<20});

        runner.run<cpu::Scan>(RunType::CPU);

        //runner.run<gpu::clpp::Scan>(RunType::CL_GPU, false); // not working
        runner.run<gpu::gpugems::NaiveScan>(RunType::CL_GPU, false);
        runner.run<gpu::gpugems::WorkEfficientScan>(RunType::CL_GPU, false);
        runner.run<gpu::gpugems::LocalScan>(RunType::CL_GPU, false);
        runner.run<gpu::apple::Scan>(RunType::CL_GPU, false);
        //runner.run<gpu::dixxi::ScanTask>(RunType::CL_GPU, false);
        runner.run<gpu::nvidia::Scan>(RunType::CL_GPU, false);

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
