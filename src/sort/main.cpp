#include <iostream>
#include <fstream>

#include "../common/Runner.h"
#include "SortPlugin.h"

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
#include "gpu/nvidia/RadixSort.h"
#include "gpu/nvidia/BitonicSort.h"
#include "gpu/dixxi/RadixSort.h"
#include "gpu/dixxi/RadixSortAtomicCounters.h"
#include "gpu/amd_dixxi/RadixSort.h"
#include "gpu/gpugems/OddEvenTransition.h"

using namespace std;

int main()
{
    try
    {
        Runner<cl_uint, SortPlugin> runner(1, { 1<<10, 1<<15, 1<<17, 1<<19, 1<<20, 1<<21, 1<<22, 1<<23, 1<<24 });
        //Runner<cl_uint, SortPlugin> runner(1, { 1<<17 });

        /*runner.run<cpu::Quicksort>(RunType::CPU);
        runner.run<cpu::QSort>(RunType::CPU);
        runner.run<cpu::STLSort>(RunType::CPU);
        //runner.run<cpu::TimSort>(RunType::CPU);
        runner.run<cpu::amd::RadixSort>(RunType::CPU);

        //runner.run<gpu::bealto::ParallelSelectionSort>(RunType::CL_GPU, true);
        //runner.run<gpu::bealto::ParallelSelectionSortLocal>(RunType::CL_GPU, true);
        //runner.run<gpu::bealto::ParallelSelectionSortBlocks>(RunType::CL_GPU, true);
        //runner.run<gpu::bealto::ParallelBitonicSortLocal>(RunType::CL_GPU, true);
        //runner.run<gpu::bealto::ParallelBitonicSortLocalOptim>(RunType::CL_GPU, true);
        //runner.run<gpu::bealto::ParallelBitonicSortA>(RunType::CL_GPU, false);
        //runner.run<gpu::bealto::ParallelBitonicSortB2>(RunType::CL_GPU, true);
        //runner.run<gpu::bealto::ParallelBitonicSortB4>(RunType::CL_GPU, true);
        //runner.run<gpu::bealto::ParallelBitonicSortB8>(RunType::CL_GPU, true);
        //runner.run<gpu::bealto::ParallelBitonicSortB16>(RunType::CL_GPU, false);
        runner.run<gpu::bealto::ParallelBitonicSortC>(RunType::CL_GPU, false);
        //runner.run<gpu::bealto::ParallelMergeSort>(RunType::CL_GPU, true);

        //runner.run<gpu::clpp::RadixSort>(RunType::CL_GPU, true); // not working

        runner.run<gpu::libcl::RadixSort>(RunType::CL_GPU, false);

        runner.run<gpu::amd::BitonicSort>(RunType::CL_GPU, false);
        runner.run<gpu::amd::RadixSort>(RunType::CL_GPU, false); // crashes on large arrays
        runner.run<gpu::amd_dixxi::RadixSort>(RunType::CL_GPU, false);*/

        //runner.run<gpu::nvidia::RadixSort>(RunType::CL_GPU, false);
        runner.run<gpu::nvidia::BitonicSort>(RunType::CL_GPU, false);

        //runner.run<gpu::dixxi::RadixSort>(RunType::CL_GPU, false);
        //runner.run<gpu::dixxi::RadixSortAtomicCounters>(RunType::CL_GPU, false);

        //runner.run<gpu::gpugems::OddEvenTransition>(RunType::CL_GPU, false);

        runner.writeStats("stats.csv");
        runner.writeGPUDeviceInfo("gpuinfo.csv");
    }
    catch(OpenCLException& e)
    {
        cerr << e.what() << endl;
        return 1;
    }

    return 0;
}
