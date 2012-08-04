// N threads, WG is workgroup size. Sort WG input blocks in each workgroup.
__kernel void ParallelMergeSort(__global const int * in,__global int * out,__local int * aux)
{
    int i = get_local_id(0); // index in workgroup
    int wg = get_local_size(0); // workgroup size = block size, power of 2

    // Move IN, OUT to block start
    int offset = get_group_id(0) * wg;
    in += offset;
    out += offset;

    // Load block in AUX[WG]
    aux[i] = in[i];
    barrier(CLK_LOCAL_MEM_FENCE); // make sure AUX is entirely up to date

    // Now we will merge sub-sequences of length 1,2,...,WG/2
    for (int length=1; length<wg; length<<=1)
    {
        int iData = aux[i];
        int iKey = iData;
        int ii = i & (length-1);  // index in our sequence in 0..length-1
        int sibling = (i - ii) ^ length; // beginning of the sibling sequence
        int pos = 0;
        for (int inc=length; inc>0; inc>>=1) // increment for dichotomic search
        {
            int j = sibling+pos+inc-1;
            int jKey = aux[j];
            bool smaller = (jKey < iKey) || ( jKey == iKey && j < i );
            pos += (smaller)?inc:0;
            pos = min(pos,length);
        }
        int bits = 2*length-1; // mask for destination
        int dest = ((ii + pos) & bits) | (i & ~bits); // destination index in merged sequence
        barrier(CLK_LOCAL_MEM_FENCE);
        aux[dest] = iData;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write output
    out[i] = aux[i];
}
