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
#include "gpu/amd/BitonicSort.h"
#include "gpu/amd/RadixSort.h"

using namespace std;
using namespace amd;

int main()
{
    const size_t size = 256 * 256 * 64;

    RUN(Quicksort, size, int);
    //RUN(QSort);
    //RUN(STLSort);
    //RUN(TimSort);

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

        //RUN_CL(ParallelSelectionSort);
        //RUN_CL(ParallelSelectionSortLocal);
        //RUN_CL(ParallelSelectionSortBlocks);
        //RUN_CL(ParallelBitonicSortLocal);
        //RUN_CL(ParallelBitonicSortLocalOptim);
        //RUN_CL(ParallelBitonicSortA);
        //RUN_CL(ParallelBitonicSortB2);
        //RUN_CL(ParallelBitonicSortB4);
        //RUN_CL(ParallelBitonicSortB8, size, int);
        RUN_CL(ParallelBitonicSortB16, size, int);
        //RUN_CL(ParallelBitonicSortC, size, int);
        //RUN_CL(ParallelMergeSort);

        RUN_CL(libcl::RadixSort, size, int);

        RUN_CL(amd::BitonicSort, size, int);
        RUN_CL(amd::RadixSort, size, int);

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
