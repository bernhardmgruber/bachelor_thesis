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

        runner.printCLInfo();

        size_t range[] = {3, 10, 30, 50, 100, 200, 300 };
        size_t length = sizeof(range) / sizeof(size_t);

        runner.printRange<cpu::Mult>(RunType::CPU, range, length);
        runner.printRange<cpu::MultThreads>(RunType::CPU, range, length);
        runner.printRange<gpu::book::Mult>(RunType::CL_GPU, range, length, false);

        runner.writeStats("stats.csv");
    }
    catch(OpenCLException& e)
    {
        cerr << e.what() << endl;
        return 1;
    }

    return 0;
}
