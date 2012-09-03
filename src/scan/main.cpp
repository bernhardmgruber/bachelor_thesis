#include <CL/CL.h>
#include <iostream>
#include <fstream>

#include "../common/OpenCL.h"

#include "cpu/Scan.h"
#include "gpu/clpp/Scan.h"


using namespace std;

int main()
{
    const size_t size = 1024 * 1 * 64;

    RUN(cpu::Scan, size, int);

    try
    {
        OpenCL::init();
        Context* context = OpenCL::getGPUContext();
        CommandQueue* queue = context->createCommandQueue();

        //cout << endl;
        //cout << "Running on " << context->getInfoString(CL_DEVICE_NAME) << endl;
        //cout << "   " << context->getInfoSize(CL_DEVICE_GLOBAL_MEM_SIZE) << "B global mem" << endl;
        //cout << "   " << context->getInfoSize(CL_DEVICE_LOCAL_MEM_SIZE) << "B local mem" << endl;
        //cout << endl;

        RUN_CL(gpu::clpp::Scan, size, int);

        delete context;
        delete queue;
        OpenCL::cleanup();
    }
    catch(OpenCLException& e)
    {
        cerr << e.what() << endl;
        return 1;
    }

    return 0;
}
