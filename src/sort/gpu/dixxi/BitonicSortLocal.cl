#include "BitonicSort.cl"

__kernel void BitonicSortLocal(__global T* buffer, __local T* shared, uint inc, uint boxwidth)
{
    uint globalId = get_global_id(0);
    uint localId = get_local_id(0); 

    // Load block to local memory
    shared[localId * 2 + 0] = buffer[globalId * 2 + 0];
    shared[localId * 2 + 1] = buffer[globalId * 2 + 1];
    barrier(CLK_LOCAL_MEM_FENCE);

    bool asc = ((globalId << 1) & boxwidth) == 0;

    // Loop on comparison distance (between keys)
    for (; inc > 0; inc >>= 1)
    {
        uint low = localId & (inc - 1); // low order bits (below inc)
        uint i = (localId << 1) - low; // insert 0 at position inc
        uint j = i + inc; // sibling to compare

        T x0 = shared[i];
        T x1 = shared[j];

        bool swap = asc ^ (x0 < x1);
        if(swap) {
            shared[i] = x1;
            shared[j] = x0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write output
    buffer[globalId * 2 + 0] = shared[localId * 2 + 0];
    buffer[globalId * 2 + 1] = shared[localId * 2 + 1];
}
