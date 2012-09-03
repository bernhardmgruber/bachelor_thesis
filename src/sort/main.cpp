#include <CL/CL.h>
#include <iostream>
#include <fstream>

#include "Runner.h"

#include "cpu/Quicksort.h"
#include "cpu/QSort.h"
#include "cpu/STLSort.h"
#include "cpu/TimSort.h"
#include "cpu/amd/RadixSort.h"

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
#include "gpu/clpp/RadixSort.h"
#include "gpu/amd/BitonicSort.h"
#include "gpu/amd/RadixSort.h"

using namespace std;

int main()
{
    const size_t size = 256 * 256 * 16;

    try
    {
        Runner<int, size> runner;

        runner.run<cpu::Quicksort>();
        runner.run<cpu::QSort>();
        runner.run<cpu::STLSort>();
        runner.run<cpu::TimSort>();
        runner.run<cpu::amd::RadixSort>();


        //cout << endl;
        //cout << "Running on " << context->getInfoString(CL_DEVICE_NAME) << endl;
        //cout << "   " << context->getInfoSize(CL_DEVICE_GLOBAL_MEM_SIZE) << "B global mem" << endl;
        //cout << "   " << context->getInfoSize(CL_DEVICE_LOCAL_MEM_SIZE) << "B local mem" << endl;
        //cout << endl;

        //RUN_CL(gpu::bealto::ParallelSelectionSort, size, int);
        //RUN_CL(gpu::bealto::ParallelSelectionSortLocal, size, int);
        //RUN_CL(gpu::bealto::ParallelSelectionSortBlocks, size, int);
        //RUN_CL(gpu::bealto::ParallelBitonicSortLocal, size, int);
        //RUN_CL(gpu::bealto::ParallelBitonicSortLocalOptim, size, int);
        //RUN_CL(gpu::bealto::ParallelBitonicSortA, size, int);
        //RUN_CL(gpu::bealto::ParallelBitonicSortB2, size, int);
        //RUN_CL(gpu::bealto::ParallelBitonicSortB4, size, int);
        //RUN_CL(gpu::bealto::ParallelBitonicSortB8, size, int);
        //RUN_CL(gpu::bealto::ParallelBitonicSortB16, size, int);
        //RUN_CL(gpu::bealto::ParallelBitonicSortC, size, int);
        //RUN_CL(gpu::bealto::ParallelMergeSort);

        //RUN_CL(gpu::clpp::RadixSort, size, int); // not working

        //RUN_CL(gpu::libcl::RadixSort, size, int);

        //RUN_CL(gpu::amd::BitonicSort, size, int);
        //RUN_CL(gpu::amd::RadixSort, size, int);
    }
    catch(OpenCLException& e)
    {
        cerr << e.what() << endl;
        return 1;
    }

    return 0;
}
