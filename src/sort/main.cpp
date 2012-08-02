#include <CL/CL.h>
#include <iostream>
#include <fstream>

#include "OpenCL.h"

#include "cpu/Quicksort.h"
#include "cpu/QSort.h"
#include "cpu/STLSort.h"
#include "cpu/TimSort.h"

#include "gpu/bealto/ParallelSelectionSort.h"
#include "gpu/bealto/ParallelSelectionSortLocal.h"
#include "gpu/bealto/ParallelSelectionSortBlocks.h"
#include "gpu/bealto/ParallelBitonicSortLocal.h"
#include "gpu/bealto/ParallelBitonicSortA.h"

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

    const size_t size = 256 * 256;
    SortingAlgorithm<int, size>* alg;

    RUN(Quicksort);
    //RUN(QSort);
    //RUN(STLSort);
    //RUN(TimSort);

    try
    {
        OpenCL::init();
        Context* context = OpenCL::getGPUContext();
        CommandQueue* queue = context->createCommandQueue();

        //RUN_GPU(ParallelSelectionSort);
        RUN_GPU(ParallelSelectionSortLocal);
        //RUN_GPU(ParallelSelectionSortBlocks);
        RUN_GPU(ParallelBitonicSortLocal);
        RUN_GPU(ParallelBitonicSortA);

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
