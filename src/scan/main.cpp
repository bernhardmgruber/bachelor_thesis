#include <CL/CL.h>
#include <iostream>
#include <fstream>

#include "../common/Runner.h"

#include "cpu/Scan.h"
#include "gpu/clpp/Scan.h"
#include "gpu/gpugems/NaiveScan.h"
#include "gpu/gpugems/WorkEfficientScan.h"
#include "gpu/gpugems/LocalScan.h"
#include "gpu/apple/Scan.h"

using namespace std;

int main()
{
    const size_t size = 1024 * 1;

    try
    {
        Runner<int, size> runner;

        runner.printCLInfo();

        runner.run<cpu::Scan>();

        //runner.runCLGPU<gpu::clpp::Scan>(true); // not working
        runner.runCLGPU<gpu::gpugems::NaiveScan>(false);
        runner.runCLGPU<gpu::gpugems::WorkEfficientScan>(false);
        runner.runCLGPU<gpu::gpugems::LocalScan>(false);
        runner.runCLGPU<gpu::apple::Scan>(false);
    }
    catch(OpenCLException& e)
    {
        cerr << e.what() << endl;
        return 1;
    }

    return 0;
}
