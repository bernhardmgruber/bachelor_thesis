#include <CL/CL.h>
#include <iostream>
#include <fstream>

#include "../common/Runner.h"
#include "MatrixPlugin.h"

#include "cpu/Mult.h"
#include "cpu/MultThreads.h"
#include "gpu/book/Mult.h"

using namespace std;

int main()
{
    try
    {
        Runner<int, MatrixPlugin> runner;

        //runner.printCLInfo();

        //size_t range[] = {1, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000 };
        size_t range[] = {1, 25, 50, 75, 100, 150, 200, 250, 300, 400 };
        size_t length = sizeof(range) / sizeof(size_t);

        runner.printRange<cpu::Mult>(RunType::CPU, range, length);
        runner.printRange<cpu::MultThreads>(RunType::CPU, range, length);
        runner.printRange<gpu::book::Mult>(RunType::CL_GPU, range, length, false);
        runner.printRange<gpu::book::Mult>(RunType::CL_CPU, range, length, false);

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
