#include <CL/CL.h>
#include <iostream>
#include <fstream>

#include "cpu/Quicksort.h"
#include "cpu/QSort.h"
#include "cpu/STLSort.h"
#include "cpu/TimSort.h"

using namespace std;

#define RUN(Cls)                \
    alg = new Cls<int, size>(); \
    alg->runTest();             \
    delete alg;

int main()
{
    const size_t size = 10000000;
    SortingAlgorithm<int, size>* alg;

    RUN(Quicksort);
    //RUN(QSort);
    //RUN(STLSort);
    RUN(TimSort);

    return 0;
}
