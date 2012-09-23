__kernel void ParallelSelectionSortLocal(__global const T * in,__global T * out,__local T * aux)
{
    int id = get_local_id(0); // index in workgroup
    int wg = get_local_size(0); // workgroup size = block size

    // Move IN, OUT to block start
    int offset = get_group_id(0) * wg;
    in += offset;
    out += offset;

    // Load block in aus[wg]
    T element = in[id];
    aux[id] = element;

    barrier(CLK_LOCAL_MEM_FENCE);

    // Find output position of element
    int pos = 0;
    for (int j = 0; j < wg; j++)
    {
        T jKey = aux[j];
        bool smaller = (jKey < element) || ( jKey == element && j < id ); // in[j] < in[id] ?
        pos += (smaller) ? 1 : 0;
    }

    // Store output
    out[pos] = element;
}
