
__kernel void ParallelSelection(__global const data_t * in,__global data_t * out)
{
    int i = get_global_id(0); // current thread
    int n = get_global_size(0); // input size
    data_t iData = in[i];
    uint iKey = getKey(iData);
    // Compute position of in[i] in output
    int pos = 0;
    for (int j=0; j<n; j++)
    {
        uint jKey = getKey(in[j]); // broadcasted
        bool smaller = (jKey < iKey) || (jKey == iKey && j < i); // in[j] < in[i] ?
        pos += (smaller) ? 1 : 0;
    }
    out[pos] = iData;
}
