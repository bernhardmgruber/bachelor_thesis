
// N threads, WG is workgroup size. Sort WG input blocks in each workgroup.
__kernel void ParallelBitonicSortLocal(__global const T * in,__global T * out,__local T * aux)
{
    int id = get_local_id(0); // index in workgroup
    int wg = get_local_size(0); // workgroup size = block size, power of 2

    // Move IN, OUT to block start
    int offset = get_group_id(0) * wg;
    in += offset;
    out += offset;

    // Load block in AUX[WG]
    aux[id] = in[id];
    barrier(CLK_LOCAL_MEM_FENCE); // make sure AUX is entirely up to date

    // Loop on sorted sequence length
    for (int length=1; length<wg; length<<=1)
    {
        bool direction = ((id & (length<<1)) != 0); // direction of sort: 0=asc, 1=desc
        // Loop on comparison distance (between keys)
        for (int inc=length; inc>0; inc>>=1)
        {
            int j = id ^ inc; // sibling to compare
            T iData = aux[id];
            T jData = aux[j];
            bool smaller = (jData < iData) || ( jData == iData && j < id );
            bool swap = smaller ^ (j < id) ^ direction;
            barrier(CLK_LOCAL_MEM_FENCE);
            aux[id] = (swap)?jData:iData;
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    // Write output
    out[id] = aux[id];
}
