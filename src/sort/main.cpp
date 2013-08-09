#include <iostream>
#include <fstream>
#include <array>

#include "../common/Runner.h"
#include "SortPlugin.h"

#include "cpu/Quicksort.h"
#include "cpu/QSort.h"
#include "cpu/STLSort.h"
#include "cpu/TimSort.h"
#include "cpu/amd/RadixSort.h"
#include "cpu/stereopsis/radixsort.h"

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
#include "gpu/nvidia/RadixSort.h"
#include "gpu/nvidia/BitonicSort.h"
#include "gpu/dixxi/RadixSort.h"
#include "gpu/dixxi/RadixSortAtomicCounters.h"
#include "gpu/dixxi/BitonicSort.h"
#include "gpu/dixxi/BitonicSortFusion.h"
#include "gpu/dixxi/BitonicSortLocal.h"
#include "gpu/amd_dixxi/RadixSort.h"
#include "gpu/gpugems/OddEvenTransition.h"

using namespace std;

#define MAX_POWER_OF_TWO 26
#define RESOLUTION 5

int main()
{
    try
    {
        //set<size_t> sizes;

        //for(int i = 1 * RESOLUTION; i <= MAX_POWER_OF_TWO * RESOLUTION; i++)
        //{
        //    size_t s = (size_t)pow(2.0, (double)i / (double)RESOLUTION);
        //    sizes.insert(s);
        //}

        //array<size_t, 13> sizes = { 1<<10, 1<<15, 1<<17, 1<<19, 1<<20, 1<<21, 1<<22, 1<<23, 1<<24, 1<<25, 1<<26 };
        array<size_t, 1> sizes = { 1<<26 };
        Runner<cl_uint, SortPlugin> runner(1, sizes.begin(), sizes.end());

        runner.writeGPUDeviceInfo("gpuinfo.csv");

        runner.start("stats.csv");

        //runner.run<cpu::Quicksort>();
        //runner.run<cpu::QSort>();
        runner.run<cpu::STLSort>();
        //runner.run<cpu::TimSort>();
        //runner.run<cpu::amd::RadixSort>();
        //runner.run<cpu::stereopsis::RadixSort>();

        //runner.run<gpu::bealto::ParallelSelectionSort>(CLRunType::GPU);
        //runner.run<gpu::bealto::ParallelSelectionSortLocal>(CLRunType::GPU);
        //runner.run<gpu::bealto::ParallelSelectionSortBlocks>(CLRunType::GPU);
        //runner.run<gpu::bealto::ParallelBitonicSortLocal>(CLRunType::GPU);
        //runner.run<gpu::bealto::ParallelBitonicSortLocalOptim>(CLRunType::GPU);
        //runner.run<gpu::bealto::ParallelBitonicSortA>(CLRunType::GPU);
        //runner.run<gpu::bealto::ParallelBitonicSortB2>(CLRunType::GPU);
        //runner.run<gpu::bealto::ParallelBitonicSortB4>(CLRunType::GPU);
        //runner.run<gpu::bealto::ParallelBitonicSortB8>(CLRunType::GPU);
        //runner.run<gpu::bealto::ParallelBitonicSortB16>(CLRunType::GPU);
        //runner.run<gpu::bealto::ParallelBitonicSortC>(CLRunType::GPU);
        //runner.run<gpu::bealto::ParallelMergeSort>(CLRunType::GPU);

        //runner.run<gpu::clpp::RadixSort>(CLRunType::GPU); // not working

        //runner.run<gpu::libcl::RadixSort>(CLRunType::GPU);

        //runner.run<gpu::amd::BitonicSort>(CLRunType::GPU);
        //runner.run<gpu::amd::RadixSort>(CLRunType::GPU); // crashes on large arrays
        //runner.run<gpu::amd_dixxi::RadixSort>(CLRunType::GPU);

        //runner.run<gpu::nvidia::RadixSort>(CLRunType::GPU);
        //runner.run<gpu::nvidia::BitonicSort>(CLRunType::GPU);

        //runner.run<gpu::dixxi::RadixSort>(CLRunType::GPU);
        //runner.run<gpu::dixxi::RadixSortAtomicCounters>(CLRunType::GPU);
        runner.run<gpu::dixxi::BitonicSort>(CLRunType::GPU);
        //runner.run<gpu::dixxi::BitonicSortFusion>(CLRunType::GPU);
        runner.run<gpu::dixxi::BitonicSortLocal>(CLRunType::GPU);

        //runner.run<gpu::gpugems::OddEvenTransition>(CLRunType::GPU);

        //runner.writeGPUDeviceInfo("gpuinfo.csv");

        runner.finish();
    }
    catch(const exception& e)
    {
        cerr << e.what() << endl;
    }

    getchar();

    return 0;
}
