#include <CL/cl.h>
#include <iostream>
#include <fstream>

#include "../common/Runner.h"
#include "MatrixPlugin.h"

#include "cpu/dixxi/Mult.h"
#include "cpu/dixxi/MultThreads.h"
#include "cpu/cblas/Mult.h"

#include "gpu/dixxi/Mult.h"
#include "gpu/dixxi/MultImage.h"
#include "gpu/dixxi/MultHybrid.h"
#include "gpu/amdblas/Mult.h"
#include "gpu/amd/MultTile.h"
#include "gpu/amd/MultTileLocal.h"

using namespace std;

int main()
{
    try
    {
        Runner<float, MatrixPlugin> runner;

        //runner.printCLInfo();

        size_t range[] = { 1, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000};//, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000 };
        //size_t range[] = { 50 };
        size_t length = sizeof(range) / sizeof(size_t);

        //runner.printRange<cpu::dixxi::Mult>(RunType::CPU, range, length);
        //runner.printRange<cpu::dixxi::MultThreads>(RunType::CPU, range, length);
        runner.printRange<cpu::cblas::Mult>(RunType::CPU, range, length);

        runner.printRange<gpu::dixxi::Mult>(RunType::CL_GPU, range, length, false);
        //runner.printRange<gpu::dixxi::MultImage>(RunType::CL_GPU, range, length, false);
        //runner.printRange<gpu::dixxi::MultHybrid>(RunType::CL_GPU, range, length, false);

        //runner.printRange<gpu::amdblas::Mult>(RunType::CL_GPU, range, length, false);

        runner.printRange<gpu::amd::MultTile>(RunType::CL_GPU, range, length, false);
        runner.printRange<gpu::amd::MultTileLocal>(RunType::CL_GPU, range, length, false);

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
