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

using namespace std;

int main()
{
    try
    {
        Runner<int, SortPlugin> runner;

        //runner.printCLInfo();

        //size_t range[] = {1, 10, 100, 1000, 10000, 100000, 1000000, 5000000, 10000000, 25000000, 50000000, 75000000, 100000000 };
        size_t range[] = { 1024 * 256 };
        size_t length = sizeof(range) / sizeof(size_t);

        //runner.printRange<cpu::Quicksort>(RunType::CPU, range, length);
        //runner.printRange<cpu::QSort>(RunType::CPU, range, length);
        //runner.printRange<cpu::STLSort>(RunType::CPU, range, length);
        //runner.printRange<cpu::TimSort>(RunType::CPU, range, length);
        //runner.printRange<cpu::amd::RadixSort>(RunType::CPU, range, length);

        /*runner.printRange<gpu::bealto::ParallelSelectionSort>(RunType::CL_GPU, range, length, true);
        runner.printRange<gpu::bealto::ParallelSelectionSortLocal>(RunType::CL_GPU, range, length, true);
        runner.printRange<gpu::bealto::ParallelSelectionSortBlocks>(RunType::CL_GPU, range, length, true);
        runner.printRange<gpu::bealto::ParallelBitonicSortLocal>(RunType::CL_GPU, range, length, true);
        runner.printRange<gpu::bealto::ParallelBitonicSortLocalOptim>(RunType::CL_GPU, range, length, true);*/
        runner.printRange<gpu::bealto::ParallelBitonicSortA>(RunType::CL_GPU, range, length, false);
        /*runner.printRange<gpu::bealto::ParallelBitonicSortB2>(RunType::CL_GPU, range, length, true);
        runner.printRange<gpu::bealto::ParallelBitonicSortB4>(RunType::CL_GPU, range, length, true);
        runner.printRange<gpu::bealto::ParallelBitonicSortB8>(RunType::CL_GPU, range, length, true);
        runner.printRange<gpu::bealto::ParallelBitonicSortB16>(RunType::CL_GPU, range, length, true);
        runner.printRange<gpu::bealto::ParallelBitonicSortC>(RunType::CL_GPU, range, length, true);
        runner.printRange<gpu::bealto::ParallelMergeSort>(RunType::CL_GPU, range, length, true);

        //runner.printRange<gpu::clpp::RadixSort>(RunType::CL_GPU, range, length, true); // not working

        //runner.printRange<gpu::libcl::RadixSort>(RunType::CL_GPU, range, length, true); // not working

        runner.printRange<gpu::amd::BitonicSort>(RunType::CL_GPU, range, length, true);
        runner.printRange<gpu::amd::RadixSort>(RunType::CL_GPU, range, length, true);*/

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
