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
    const size_t size = 800;

    try
    {
        Runner<int, MatrixPlugin> runner;

        runner.printCLInfo();

        runner.printRun<cpu::Mult>(size);
        runner.printRun<cpu::MultThreads>(size);
        runner.printRunCLGPU<gpu::book::Mult>(size, true);

        //runner.writeStats("stats.csv");
    }
    catch(OpenCLException& e)
    {
        cerr << e.what() << endl;
        return 1;
    }

    return 0;
}
