#include <CL/CL.h>
#include <iostream>
#include <fstream>

#include "../common/OpenCL.h"

#include "cpu/OpenSSL/aes.h"

using namespace std;

int main()
{
    const size_t size = 1024 * 1024 * 64;

    RUN(cpu::openssl::AES, size);

    try
    {
        OpenCL::init();
        Context* context = OpenCL::getGPUContext();
        CommandQueue* queue = context->createCommandQueue();


        delete context;
        delete queue;
        OpenCL::cleanup();
    }
    catch(OpenCLException& e)
    {
        cerr << e.what() << endl;
    }

    return 0;
}
