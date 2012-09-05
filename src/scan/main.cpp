#include <CL/CL.h>
#include <iostream>
#include <fstream>

#include "Runner.h"

#include "cpu/Scan.h"
#include "gpu/clpp/Scan.h"


using namespace std;

int main()
{
    const size_t size = 1024 * 1 * 64;

    try
    {
        Runner<int, size> runner;

        runner.printCLInfo();

        runner.run<cpu::Scan>();

        // runner.runCLGPU<gpu::clpp::Scan>(true); // not working
    }
    catch(OpenCLException& e)
    {
        cerr << e.what() << endl;
        return 1;
    }

    return 0;
}
