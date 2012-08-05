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
#include "gpu/bealto/ParallelBitonicSortB8.h"
#include "gpu/bealto/ParallelBitonicSortB16.h"
#include "gpu/bealto/ParallelBitonicSortC.h"
#include "gpu/bealto/ParallelMergeSort.h"
#include "gpu/libCL/RadixSort.h"

using namespace std;

int main()
{
    const size_t size = 256 * 256 * 32;

    RUN(Quicksort, size, int);
    //RUN(QSort);
    //RUN(STLSort);
    //RUN(TimSort);

    try
    {
        OpenCL::init();
        Context* context = OpenCL::getGPUContext();
        CommandQueue* queue = context->createCommandQueue();

        //RUN_CL(ParallelSelectionSort);
        //RUN_CL(ParallelSelectionSortLocal);
        //RUN_CL(ParallelSelectionSortBlocks);
        //RUN_CL(ParallelBitonicSortLocal);
        //RUN_CL(ParallelBitonicSortLocalOptim);
        //RUN_CL(ParallelBitonicSortA);
        //RUN_CL(ParallelBitonicSortB2);
        //RUN_CL(ParallelBitonicSortB4);
        //RUN_CL(ParallelBitonicSortB8);
        //RUN_CL(ParallelBitonicSortB16);
        //RUN_CL(ParallelBitonicSortC, size, int);
        //RUN_CL(ParallelMergeSort);

        RUN_CL(libcl::RadixSort, size, int);

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
