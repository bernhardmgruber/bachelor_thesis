#include <CL/CL.h>
#include <iostream>
#include <fstream>

#include "../common/Runner.h"
#include "MatrixVerifier.h"

#include "cpu/Mult.h"

using namespace std;

int main()
{
    const size_t size = 1024;

    try
    {
        Runner<int, size, MatrixVerifier> runner;

        runner.printCLInfo();

        runner.run<cpu::Mult>();
    }
    catch(OpenCLException& e)
    {
        cerr << e.what() << endl;
        return 1;
    }

    return 0;
}
