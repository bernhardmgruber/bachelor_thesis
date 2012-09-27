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
    const size_t size = 500;

    try
    {
        Runner<int, MatrixPlugin> runner;

        runner.printCLInfo();

        runner.printOnce<cpu::Mult, RunType::CPU>(size);
        runner.printOnce<cpu::MultThreads, RunType::CPU>(size);
        runner.printOnce<gpu::book::Mult, RunType::CL_GPU>(size);

        runner.writeStats("stats.csv");
    }
    catch(OpenCLException& e)
    {
        cerr << e.what() << endl;
        return 1;
    }

    return 0;
}
