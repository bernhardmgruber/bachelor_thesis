#include <CL/cl.h>
#include <iostream>
#include <fstream>

#include "../common/Runner.h"
#include "MeshTransformPlugin.h"

#include "cpu/dixxi/Transform.h"
#include "cpu/dixxi/TransformMulti.h"
#include "gpu/dixxi/Transform.h"
#include "gpu/dixxi/TransformBuiltIn.h"

using namespace std;

int main()
{
    try
    {
        Runner<float, MeshTransformPlugin> runner(5, { 1<<23, 1<<25 });

        runner.run<cpu::dixxi::Transform>(RunType::CPU);
        runner.run<cpu::dixxi::TransformMulti>(RunType::CPU);

        runner.run<gpu::dixxi::Transform>(RunType::CL_GPU, false);
        runner.run<gpu::dixxi::TransformBuiltIn>(RunType::CL_GPU, false);

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
