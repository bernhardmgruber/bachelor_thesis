#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <array>

#include "../common/Runner.h"
#include "MatrixPlugin.h"

//#include "cpu/dixxi/Mult.h"
//#include "cpu/dixxi/MultThreads.h"
//#include "cpu/cblas/Mult.h"
//
#include "gpu/dixxi/Mult1D.h"
#include "gpu/dixxi/Mult2D.h"
//#include "gpu/dixxi/Mult2DCoalesced.h"
#include "gpu/dixxi/MultBlock.h"
//#include "gpu/dixxi/MultImage.h"
//#include "gpu/dixxi/MultHybrid.h"
#include "gpu/dixxi/MultTileAMD.h"
#include "gpu/dixxi/MultTileAMDArr.h"
#include "gpu/dixxi/MultTileLocalAMD.h"
//#include "gpu/amdblas/Mult.h"
#include "gpu/amd/MultTile.h"
#include "gpu/amd/MultTileLocal.h"
//#include "gpu/nvidia/Mult.h"
#include "gpu/preso/MultLocal.h"

using namespace std;

int main()
{ 
    try
    {
        //vector<size_t> sizes = { 1, 25, 50, 75, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000 };
        array<size_t, 1> sizes = { 4000 };

        //vector<size_t> sizes;
        //sizes.push_back(1);
        //sizes.push_back(25);
        //sizes.push_back(50);
        //sizes.push_back(75);
        //for(int i = 1; i <= 40; i++)
        //	sizes.push_back(i * 100);

        //Runner<float, MatrixPlugin> runner(3, { 1, 25, 50, 75, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000 }, false);
        Runner<float, MatrixPlugin> runner(1, sizes.begin(), sizes.end(), false);

        //runner.writeGPUDeviceInfo("gpuinfo.csv");

        //        runner.run<cpu::dixxi::Mult>();
        //        runner.writeStats("stats.csv");
        //        runner.run<cpu::dixxi::MultThreads>();
        //        runner.writeStats("stats.csv");
        //        runner.run<cpu::cblas::Mult>();
        //        runner.writeStats("stats.csv");

        //runner.run<gpu::dixxi::Mult1D>(CLRunType::GPU);
        //        runner.writeStats("stats.csv");
        //        runner.run<gpu::dixxi::Mult2D>(CLRunType::GPU);
        //        runner.writeStats("stats.csv");
        //        runner.run<gpu::dixxi::Mult2DCoalesced>(CLRunType::GPU);
        //        runner.writeStats("stats.csv");
        //        runner.run<gpu::dixxi::MultBlock>(CLRunType::GPU);
        //        runner.writeStats("stats.csv");
        //        runner.run<gpu::dixxi::MultImage>(CLRunType::GPU);
        //        runner.writeStats("stats.csv");
        //        runner.run<gpu::dixxi::MultHybrid>(CLRunType::GPU);
        //        runner.writeStats("stats.csv");
        runner.run<gpu::dixxi::MultTileAMD>(CLRunType::GPU);
        //        runner.writeStats("stats.csv");
        runner.run<gpu::dixxi::MultTileAMDArr>(CLRunType::GPU);
        //        runner.writeStats("stats.csv");
        runner.run<gpu::dixxi::MultTileLocalAMD>(CLRunType::GPU);
        //        runner.writeStats("stats.csv");
        //
        //runner.run<gpu::amdblas::Mult>(CLRunType::GPU); // crashes in x64 on invocation, maybe compiler issue? samples also crash, when compiled with gcc, provided binaries of samples work

        runner.run<gpu::amd::MultTile>(CLRunType::GPU);
        //runner.writeStats("stats.csv");
        runner.run<gpu::amd::MultTileLocal>(CLRunType::GPU);
        //runner.writeStats("stats.csv");

        //        runner.run<gpu::nvidia::Mult>(CLRunType::GPU);
        //        runner.writeStats("stats.csv");

        //runner.run<gpu::preso::MultLocal>(CLRunType::GPU);
    }
    catch(const exception& e)
    {
        cerr << e.what() << endl;
    }

    getchar();

    return 0;
}
