
#include "ParallelBitonicSortB16.cl"

__kernel void ParallelBitonicSortC4(__global T * data, int inc0, int dir, __local T * aux)
{
    int t = get_global_id(0); // thread index
    int wgBits = 4*get_local_size(0) - 1; // bit mask to get index in local memory AUX (size is 4*WG)
    int inc,low,i;
    bool reverse;
    T x[4];

    // First iteration, global input, local output
    inc = inc0>>1;
    low = t & (inc - 1); // low order bits (below INC)
    i = ((t - low) << 2) + low; // insert 00 at position INC
    reverse = ((dir & i) == 0); // asc/desc order
    for (int k=0; k<4; k++)
        x[k] = data[i+k*inc];

    B4V(x,0);

    for (int k=0; k<4; k++)
        aux[(i+k*inc) & wgBits] = x[k];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Internal iterations, local input and output
    for ( ; inc>1; inc>>=2)
    {
        low = t & (inc - 1); // low order bits (below INC)
        i = ((t - low) << 2) + low; // insert 00 at position INC
        reverse = ((dir & i) == 0); // asc/desc order
        for (int k=0; k<4; k++)
            x[k] = aux[(i+k*inc) & wgBits];

        B4V(x,0);
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k=0; k<4; k++)
            aux[(i+k*inc) & wgBits] = x[k];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Final iteration, local input, global output, INC=1
    i = t << 2;
    reverse = ((dir & i) == 0); // asc/desc order
    for (int k=0; k<4; k++)
        x[k] = aux[(i+k) & wgBits];
    B4V(x,0);
    for (int k=0; k<4; k++)
    data[i+k] = x[k];
}
