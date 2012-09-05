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
    const size_t size = 256 * 8;

    try
    {
        Runner<int, size> runner;

        runner.printCLInfo();

        runner.run<cpu::Quicksort>();
        runner.run<cpu::QSort>();
        runner.run<cpu::STLSort>();
        runner.run<cpu::TimSort>();
        runner.run<cpu::amd::RadixSort>();

        runner.runCLGPU<gpu::bealto::ParallelSelectionSort>(true);
        runner.runCLGPU<gpu::bealto::ParallelSelectionSortLocal>(true);
        runner.runCLGPU<gpu::bealto::ParallelSelectionSortBlocks>(true);
        runner.runCLGPU<gpu::bealto::ParallelBitonicSortLocal>(true);
        runner.runCLGPU<gpu::bealto::ParallelBitonicSortLocalOptim>(true);
        runner.runCLGPU<gpu::bealto::ParallelBitonicSortA>(true);
        runner.runCLGPU<gpu::bealto::ParallelBitonicSortB2>(true);
        runner.runCLGPU<gpu::bealto::ParallelBitonicSortB4>(true);
        runner.runCLGPU<gpu::bealto::ParallelBitonicSortB8>(true);
        runner.runCLGPU<gpu::bealto::ParallelBitonicSortB16>(true);
        runner.runCLGPU<gpu::bealto::ParallelBitonicSortC>(true);
        runner.runCLGPU<gpu::bealto::ParallelMergeSort>(true);

        //runner.runCLGPU<gpu::clpp::RadixSort>(true); // not working

        //runner.runCLGPU<gpu::libcl::RadixSort>(true); // not working

        runner.runCLGPU<gpu::amd::BitonicSort>(true);
        runner.runCLGPU<gpu::amd::RadixSort>(true);
    }
    catch(OpenCLException& e)
    {
        cerr << e.what() << endl;
        return 1;
    }

    return 0;
}
