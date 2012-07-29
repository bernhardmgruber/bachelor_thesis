#include <CL/CL.h>
#include <iostream>
#include <fstream>

#include "OpenCL.h"

#include "cpu/Quicksort.h"
#include "cpu/QSort.h"
#include "cpu/STLSort.h"
#include "cpu/TimSort.h"

#include "gpu/bealto/ParallelSelectionSort.h"

using namespace std;

#define RUN(Cls)                \
    alg = new Cls<int, size>(); \
    alg->runTest();             \
    delete alg;

#define RUN_GPU(Cls)                          \
    alg = new Cls<int, size>(context, queue); \
    alg->runTest();                           \
    delete alg;

int main()
{
    const size_t size = 10000000;
    SortingAlgorithm<int, size>* alg;

    //RUN(Quicksort);
    //RUN(QSort);
    //RUN(STLSort);
    //RUN(TimSort);

    OpenCL::init();
    Context* context = OpenCL::getGPUContext();
    CommandQueue* queue = context->createCommandQueue();

    RUN_GPU(ParallelSelectionSort);

    delete context;
    delete queue;
    OpenCL::cleanup();

    return 0;
}
