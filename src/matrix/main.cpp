#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <array>

#include "../common/Runner.h"
#include "MatrixPlugin.h"

#include "cpu/dixxi/Mult.h"
#include "cpu/dixxi/MultThreads.h"
#include "cpu/cblas/Mult.h"

#include "gpu/dixxi/Mult1D.h"
#include "gpu/dixxi/Mult2D.h"
#include "gpu/dixxi/Mult2DCoalesced.h"
#include "gpu/dixxi/MultBlock.h"
#include "gpu/dixxi/MultImage.h"
#include "gpu/dixxi/MultHybrid.h"
//#include "gpu/amdblas/Mult.h"
#include "gpu/amd/MultTile.h"
#include "gpu/amd/MultTileLocal.h"
#include "gpu/nvidia/Mult.h"

using namespace std;

int main()
{
    try
    {
        //size_t arr[] = { 1, 25, 50, 75, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 0 };
        array<size_t, 44> sizes = { 1, 25, 50, 75, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000 };

        //vector<size_t> sizes;
        //sizes.push_back(1);
        //sizes.push_back(25);
        //sizes.push_back(50);
        //sizes.push_back(75);
        //for(int i = 1; i <= 40; i++)
        //	sizes.push_back(i * 100);

        //Runner<float, MatrixPlugin> runner(3, { 1, 25, 50, 75, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000 }, false);
        Runner<float, MatrixPlugin> runner(3, sizes.begin(), sizes.end(), false);

        runner.writeGPUDeviceInfo("gpuinfo.csv");

        runner.run<cpu::dixxi::Mult>(RunType::CPU);
        runner.writeStats("stats.csv");
        runner.run<cpu::dixxi::MultThreads>(RunType::CPU);
        runner.writeStats("stats.csv");
        runner.run<cpu::cblas::Mult>(RunType::CPU);
        runner.writeStats("stats.csv");

        runner.run<gpu::dixxi::Mult1D>(RunType::CL_GPU, false);
        runner.writeStats("stats.csv");
        runner.run<gpu::dixxi::Mult2D>(RunType::CL_GPU, false);
        runner.writeStats("stats.csv");
        runner.run<gpu::dixxi::Mult2DCoalesced>(RunType::CL_GPU, false);
        runner.writeStats("stats.csv");
        runner.run<gpu::dixxi::MultBlock>(RunType::CL_GPU, false);
        runner.writeStats("stats.csv");
        runner.run<gpu::dixxi::MultImage>(RunType::CL_GPU, false);
        runner.writeStats("stats.csv");
        runner.run<gpu::dixxi::MultHybrid>(RunType::CL_GPU, false);
        runner.writeStats("stats.csv");

        //runner.run<gpu::amdblas::Mult>(RunType::CL_GPU, false); // crashes in x64 on invocation, maybe compiler issue? samples also crash, when compiled with gcc, provided binaries of samples work

        runner.run<gpu::amd::MultTile>(RunType::CL_GPU, false);
        runner.writeStats("stats.csv");
        runner.run<gpu::amd::MultTileLocal>(RunType::CL_GPU, false);
        runner.writeStats("stats.csv");

        runner.run<gpu::nvidia::Mult>(RunType::CL_GPU, false);
        runner.writeStats("stats.csv");
    }
    catch(OpenCLException& e)
    {
        cerr << e.what() << endl;
    }

    return 0;
}
