#ifndef BLOCK_FACTOR
#define BLOCK_FACTOR 1
#endif

// One thread per record, local memory size AUX is BLOCK_FACTOR * workgroup size keys
__kernel void ParallelSelection_Blocks(__global const int* in, __global int* out, __local int* aux)
{
    int id = get_global_id(0); // current thread
    int n = get_global_size(0); // input size
    int wg = get_local_size(0); // workgroup size
    int element = in[id]; // input record for current thread
    int blockSize = BLOCK_FACTOR * wg; // block size

    // Compute position of element in output
    int pos = 0;
    // Loop on blocks of size BLOCKSIZE keys (BLOCKSIZE must divide N)
    for (int j=0; j<n; j+=blockSize)
    {
        // Load BLOCKSIZE keys using all threads (BLOCK_FACTOR values per thread)
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int index=get_local_id(0); index<blockSize; index+=wg)
        {
            aux[index] = getKey(in[j+index]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Loop on all values in AUX
        for (int index=0; index<blockSize; index++)
        {
            uint jKey = aux[index]; // broadcasted, local memory
            bool smaller = (jKey < element) || ( jKey == element && (j+index) < id ); // in[j] < in[id] ?
            pos += (smaller)?1:0;
        }
    }
    out[pos] = element;
}
