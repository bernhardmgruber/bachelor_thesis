
__kernel void ParallelBitonicSortA(__global T * in, __global T * out, int inc, int dir)
{
    int id1 = get_global_id(0); // thread index
    int id2 = id1 ^ inc; // sibling to compare

    // Load values at I and J
    T element1 = in[id1];
    T element2 = in[id2];

    // Compare
    bool smaller = (element2 < element1) || (element2 == element1 && id2 < id1);
    bool swap = smaller ^ (id2 < id1) ^ ((dir & id1) != 0);

    // Store
    out[id1] = swap ? element2 : element1;
}
