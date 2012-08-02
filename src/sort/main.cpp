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
#include "gpu/bealto/ParallelBitonicSortLocalOptim.h"
#include "gpu/bealto/ParallelBitonicSortA.h"
#include "gpu/bealto/ParallelBitonicSortB2.h"
#include "gpu/bealto/ParallelBitonicSortB4.h"

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
    const size_t size = 256 * 256 * 64;
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
        //RUN_GPU(ParallelSelectionSortLocal);
        //RUN_GPU(ParallelSelectionSortBlocks);
        //RUN_GPU(ParallelBitonicSortLocal);
        //RUN_GPU(ParallelBitonicSortLocalOptim);
        //RUN_GPU(ParallelBitonicSortA);
        //RUN_GPU(ParallelBitonicSortB2);
        RUN_GPU(ParallelBitonicSortB4);

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
