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
    const size_t size = 1024 * 1 * 128;

    try
    {
        Runner<int, ScanPlugin> runner;

        runner.printCLInfo();

        runner.printOnce<cpu::Scan>(RunType::CPU, size);

        //runner.printOnce<gpu::clpp::Scan>(RunType::CL_GPU, size, false); // not working
        runner.printOnce<gpu::gpugems::NaiveScan>(RunType::CL_GPU, size, false);
        runner.printOnce<gpu::gpugems::WorkEfficientScan>(RunType::CL_GPU, size, false);
        runner.printOnce<gpu::gpugems::LocalScan>(RunType::CL_GPU, size, false);
        runner.printOnce<gpu::apple::Scan>(RunType::CL_GPU, size, false);
    }
    catch(OpenCLException& e)
    {
        cerr << e.what() << endl;
        return 1;
    }

    return 0;
}
