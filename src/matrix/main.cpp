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
#include "gpu/dixxi/MultLocal.h"
#include "gpu/dixxi/MultImage.h"
#include "gpu/dixxi/MultHybrid.h"
#include "gpu/dixxi/MultBlockAMD.h"
#include "gpu/dixxi/MultBlockAMDArr.h"
#include "gpu/dixxi/MultBlockLocalAMD.h"
#include "gpu/dixxi/MultBlockLocalAMDTransposed.h"
#include "gpu/dixxi/MultBlockLocalOneAMD.h"
#include "gpu/amdblas/Mult.h"
#include "gpu/amd/MultBlock.h"
#include "gpu/amd/MultBlockLocal.h"
#include "gpu/nvidia/MultLocal.h"
#include "gpu/preso/MultLocal.h"

#include "gpu/thesis/Mult.h"
#include "gpu/thesis/MultLocal.h"
#include "gpu/thesis/MultBlock.h"
#include "gpu/thesis/MultBlockLocal.h"

using namespace std;

int main()
{ 
    try
    {
        //vector<size_t> sizes = { 1, 25, 50, 75, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000 };
        array<size_t, 44> sizes = { 1, 25, 50, 75, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000 };

        Runner<float, MatrixPlugin> runner(3, sizes.begin(), sizes.end(), false);

        //runner.writeGPUDeviceInfo("gpuinfo.csv");

        runner.start("stats.csv");

        //runner.run<cpu::dixxi::Mult>();
        //runner.run<cpu::dixxi::MultThreads>();
        //runner.run<cpu::cblas::Mult>();

        //runner.run<gpu::dixxi::Mult1D>(CLRunType::GPU);
        //runner.run<gpu::dixxi::Mult2D>(CLRunType::GPU);
        //runner.run<gpu::dixxi::Mult2DCoalesced>(CLRunType::GPU);
        //runner.run<gpu::dixxi::MultLocal>(CLRunType::GPU);
        //runner.run<gpu::dixxi::MultImage>(CLRunType::GPU);
        //runner.run<gpu::dixxi::MultHybrid>(CLRunType::GPU);
        //runner.run<gpu::dixxi::MultBlockAMD>(CLRunType::GPU);
        //runner.run<gpu::dixxi::MultBlockAMDArr>(CLRunType::GPU);
        //runner.run<gpu::dixxi::MultBlockLocalAMD>(CLRunType::GPU);
        //runner.run<gpu::dixxi::MultBlockLocalAMDTransposed>(CLRunType::GPU);
        //runner.run<gpu::dixxi::MultBlockLocalOneAMD>(CLRunType::GPU);

        //runner.run<gpu::amdblas::Mult>(CLRunType::GPU); // crashes in x64 on invocation when compiled with gcc

        //runner.run<gpu::amd::MultBlock>(CLRunType::GPU);
        //runner.run<gpu::amd::MultBlockLocal>(CLRunType::GPU);

        //runner.run<gpu::nvidia::MultLocal>(CLRunType::GPU);

        //runner.run<gpu::preso::MultLocal>(CLRunType::GPU);

        runner.run<gpu::thesis::Mult>(CLRunType::GPU);
        runner.run<gpu::thesis::MultLocal>(CLRunType::GPU);
        runner.run<gpu::thesis::MultBlock>(CLRunType::GPU);
        runner.run<gpu::thesis::MultBlockLocal>(CLRunType::GPU);

        runner.finish();
    }
    catch(const exception& e)
    {
        cerr << e.what() << endl;
    }

    getchar();

    return 0;
}
