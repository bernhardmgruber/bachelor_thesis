
#include "ParallelBitonicSortB16.cl"

// N/2 threads, AUX[2*WG]
/*__kernel void ParallelBitonicSortC2(__global int * data,int inc0,int dir,__local int * aux)
{
    int t = get_global_id(0); // thread index
    int wgBits = 2*get_local_size(0) - 1; // bit mask to get index in local memory AUX (size is 2*WG)

    for (int inc=inc0; inc>0; inc>>=1)
    {
        int low = t & (inc - 1); // low order bits (below INC)
        int i = (t<<1) - low; // insert 0 at position INC
        bool reverse = ((dir & i) == 0); // asc/desc order
        int x0,x1;

        // Load
        if (inc == inc0)
        {
            // First iteration: load from global memory
            x0 = data[i];
            x1 = data[i+inc];
        }
        else
        {
            // Other iterations: load from local memory
            barrier(CLK_LOCAL_MEM_FENCE);
            x0 = aux[i & wgBits];
            x1 = aux[(i+inc) & wgBits];
        }

        // Sort
        ORDER(x0,x1)

        // Store
        if (inc == 1)
        {
            // Last iteration: store to global memory
            data[i] = x0;
            data[i+inc] = x1;
        }
        else
        {
            // Other iterations: store to local memory
            barrier(CLK_LOCAL_MEM_FENCE);
            aux[i & wgBits] = x0;
            aux[(i+inc) & wgBits] = x1;
        }
    }
}*/

// N/4 threads, AUX[4*WG]
/*__kernel void ParallelBitonic_C4_0(__global int * data,int inc0,int dir,__local int * aux)
{
    int t = get_global_id(0); // thread index
    int wgBits = 4*get_local_size(0) - 1; // bit mask to get index in local memory AUX (size is 4*WG)

    for (int inc=inc0>>1; inc>0; inc>>=2)
    {
        int low = t & (inc - 1); // low order bits (below INC)
        int i = ((t - low) << 2) + low; // insert 00 at position INC
        bool reverse = ((dir & i) == 0); // asc/desc order
        int x[4];

        // Load
        if (inc == inc0>>1)
        {
            // First iteration: load from global memory
            for (int k=0; k<4; k++) x[k] = data[i+k*inc];
        }
        else
        {
            // Other iterations: load from local memory
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int k=0; k<4; k++) x[k] = aux[(i+k*inc) & wgBits];
        }

        // Sort
        B4V(x,0);

        // Store
        if (inc == 1)
        {
            // Last iteration: store to global memory
            for (int k=0; k<4; k++) data[i+k*inc] = x[k];
        }
        else
        {
            // Other iterations: store to local memory
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int k=0; k<4; k++) aux[(i+k*inc) & wgBits] = x[k];
        }
    }
}*/

__kernel void ParallelBitonicSortC4(__global int * data, int inc0, int dir, __local int * aux)
{
    int t = get_global_id(0); // thread index
    int wgBits = 4*get_local_size(0) - 1; // bit mask to get index in local memory AUX (size is 4*WG)
    int inc,low,i;
    bool reverse;
    int x[4];

    // First iteration, global input, local output
    inc = inc0>>1;
    low = t & (inc - 1); // low order bits (below INC)
    i = ((t - low) << 2) + low; // insert 00 at position INC
    reverse = ((dir & i) == 0); // asc/desc order
    for (int k=0; k<4; k++) x[k] = data[i+k*inc];
    B4V(x,0);
    for (int k=0; k<4; k++) aux[(i+k*inc) & wgBits] = x[k];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Internal iterations, local input and output
    for ( ; inc>1; inc>>=2)
    {
        low = t & (inc - 1); // low order bits (below INC)
        i = ((t - low) << 2) + low; // insert 00 at position INC
        reverse = ((dir & i) == 0); // asc/desc order
        for (int k=0; k<4; k++) x[k] = aux[(i+k*inc) & wgBits];
        B4V(x,0);
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k=0; k<4; k++) aux[(i+k*inc) & wgBits] = x[k];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Final iteration, local input, global output, INC=1
    i = t << 2;
    reverse = ((dir & i) == 0); // asc/desc order
    for (int k=0; k<4; k++) x[k] = aux[(i+k) & wgBits];
    B4V(x,0);
    for (int k=0; k<4; k++) data[i+k] = x[k];
}
