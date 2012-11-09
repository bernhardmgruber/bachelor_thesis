#include <CL/cl.h>
#include <iostream>
#include <fstream>

#include "../common/Runner.h"

using namespace std;

template <typename T>
class DummyPlugin
{

};

int main()
{
    Runner<int, DummyPlugin> runner;

    try
    {
        runner.writeCPUDeviceInfo("cpuinfo.csv");
    }
    catch(OpenCLException& e)
    {
        cerr << e.what() << endl;
    }

    try
    {
        runner.writeGPUDeviceInfo("gpuinfo.csv");
    }
    catch(OpenCLException& e)
    {
        cerr << e.what() << endl;
    }

    return 0;
}
