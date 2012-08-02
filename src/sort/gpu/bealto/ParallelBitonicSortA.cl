
__kernel void ParallelBitonicSortA(__global const int * in, __global int * out, int inc, int dir)
{
    int id = get_global_id(0); // thread index
    int j = id ^ inc; // sibling to compare

    // Load values at I and J
    int element = in[id];
    int jData = in[j];

    // Compare
    bool smaller = (jData < element) || (jData == element && j < id);
    bool swap = smaller ^ (j < id) ^ ((dir & id) != 0);

    // Store
    out[id] = swap ? jData : element;
}
