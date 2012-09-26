#include <CL/CL.h>
#include <iostream>
#include <fstream>

#include "../common/Runner.h"
#include "MatrixPlugin.h"

#include "cpu/Mult.h"

using namespace std;

int main()
{
    const size_t size = 10;

    try
    {
        Runner<int, MatrixPlugin> runner;

        runner.printCLInfo();

        runner.printRun<cpu::Mult>(size);
    }
    catch(OpenCLException& e)
    {
        cerr << e.what() << endl;
        return 1;
    }

    return 0;
}
