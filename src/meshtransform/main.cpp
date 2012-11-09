#include <CL/cl.h>
#include <iostream>
#include <fstream>

#include "../common/Runner.h"
#include "MeshTransformPlugin.h"

#include "cpu/dixxi/Transform.h"
#include "cpu/dixxi/TransformMulti.h"
#include "gpu/dixxi/Transform.h"

using namespace std;

int main()
{
    try
    {
        Runner<float, MeshTransformPlugin> runner;

        size_t range[] = { 1<<15, 1<<18, 1<<20, 1<<23 };
        //size_t range[] = { 10 };
        size_t count = sizeof(range) / sizeof(size_t);

        runner.printRange<cpu::dixxi::Transform>(RunType::CPU, range, count);
        runner.printRange<cpu::dixxi::TransformMulti>(RunType::CPU, range, count);

        runner.printRange<gpu::dixxi::Transform>(RunType::CL_GPU, range, count, false);

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
