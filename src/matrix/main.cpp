#include <CL/cl.h>
#include <iostream>
#include <fstream>

#include "../common/Runner.h"
#include "MatrixPlugin.h"

#include "cpu/dixxi/Mult.h"
#include "cpu/dixxi/MultThreads.h"
#include "cpu/cblas/Mult.h"

#include "gpu/dixxi/Mult.h"
#include "gpu/dixxi/MultBlock.h"
#include "gpu/dixxi/MultImage.h"
#include "gpu/dixxi/MultHybrid.h"
#include "gpu/amdblas/Mult.h"
#include "gpu/amd/MultTile.h"
#include "gpu/amd/MultTileLocal.h"
#include "gpu/nvidia/Mult.h"

using namespace std;

int main()
{
    try
    {
        //Runner<float, MatrixPlugin> runner(3, { 1, 25, 50, 75, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000 }, false);
        Runner<float, MatrixPlugin> runner(1, { 2000 }, false);

//        runner.run<cpu::dixxi::Mult>(RunType::CPU);
//        runner.run<cpu::dixxi::MultThreads>(RunType::CPU);
//        runner.run<cpu::cblas::Mult>(RunType::CPU);
//
        runner.run<gpu::dixxi::Mult>(RunType::CL_GPU, false);
        runner.run<gpu::dixxi::MultBlock>(RunType::CL_GPU, false);
//        runner.run<gpu::dixxi::MultImage>(RunType::CL_GPU, false);
//        runner.run<gpu::dixxi::MultHybrid>(RunType::CL_GPU, false);*/
//
//        runner.run<gpu::amdblas::Mult>(RunType::CL_GPU, false); // crashes in x64 on invocation, maybe compiler issue? samples also crash, when compiled with gcc, provided binaries of samples work
//
//        runner.run<gpu::amd::MultTile>(RunType::CL_GPU, false);
//        runner.run<gpu::amd::MultTileLocal>(RunType::CL_GPU, false);
//
//        runner.run<gpu::nvidia::Mult>(RunType::CL_GPU, false);

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
