#include <CL/CL.h>
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

using namespace std;

int main()
{
    try
    {
        Runner<int, ScanPlugin> runner(false);

        //runner.printCLInfo();

        size_t range[] = {2<<10, 2<<11, 2<<12, 2<<13, 2<<14, 2<<15, 2<<16, 2<<17, 2<<18, 2<<19, 2<<20, 2<<21, 2<<22, 2<<23, 2<<24, 2<<25};
        size_t length = sizeof(range) / sizeof(size_t);

        runner.printRange<cpu::Scan>(RunType::CPU, range, length);

        //runner.printOnce<gpu::clpp::Scan>(RunType::CL_GPU, size, false); // not working
        runner.printRange<gpu::gpugems::NaiveScan>(RunType::CL_GPU, range, length, true);
        runner.printRange<gpu::gpugems::WorkEfficientScan>(RunType::CL_GPU, range, length, true);
        runner.printRange<gpu::gpugems::LocalScan>(RunType::CL_GPU, range, length, true);
        runner.printRange<gpu::apple::Scan>(RunType::CL_GPU, range, length, true);

        // also run them on the CPU
        runner.printRange<gpu::gpugems::NaiveScan>(RunType::CL_CPU, range, length, true);
        runner.printRange<gpu::gpugems::WorkEfficientScan>(RunType::CL_CPU, range, length, true);
        runner.printRange<gpu::gpugems::LocalScan>(RunType::CL_CPU, range, length, true);
        runner.printRange<gpu::apple::Scan>(RunType::CL_CPU, range, length, true);

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
