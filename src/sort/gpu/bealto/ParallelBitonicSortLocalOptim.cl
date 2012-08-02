__kernel void ParallelBitonicSortLocalOptim(__global const int * in,__global int * out,__local int * aux)
{
    int i = get_local_id(0); // index in workgroup
    int wg = get_local_size(0); // workgroup size = block size, power of 2

    // Move IN, OUT to block start
    int offset = get_group_id(0) * wg;
    in += offset;
    out += offset;

    // Load block in AUX[WG]
    int iData = in[i];
    aux[i] = iData;
    barrier(CLK_LOCAL_MEM_FENCE); // make sure AUX is entirely up to date

    // Loop on sorted sequence length
    for (int length=1; length<wg; length<<=1)
    {
        bool direction = ((i & (length<<1)) != 0); // direction of sort: 0=asc, 1=desc
        // Loop on comparison distance (between keys)
        for (int inc=length; inc>0; inc>>=1)
        {
            int j = i ^ inc; // sibling to compare
            int jData = aux[j];
            bool smaller = (jData < iData) || ( jData == iData && j < i );
            bool swap = smaller ^ (j < i) ^ direction;
            iData = swap ? jData : iData; // update iData
            barrier(CLK_LOCAL_MEM_FENCE);
            aux[i] = iData;
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    // Write output
    out[i] = iData;
}
