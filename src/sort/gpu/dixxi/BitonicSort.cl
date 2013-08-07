
__kernel void BitonicSort(__global T* data, uint dist, uint boxwidth)
{
    uint id = get_global_id(0); // thread index
    uint low = id & (dist - 1); // low order bits (below dist)
    uint i = (id << 1) - low; // insert 0 at position dist

    bool reverse = ((boxwidth & i) == 0); // test if the bit at boxwidth is set, determines asc/desc order

    data += i; // translate to first value

    // Load
    T x0 = data[0];
    T x1 = data[dist];

    // Compare
    bool swap = reverse ^ (x0 < x1);
    if(swap) {
        // Sort
        T auxa = x0;
        T auxb = x1;
        x0 = swap ? auxb : auxa;
        x1 = swap ? auxa : auxb;

        // Store
        data[0] = x0;
        data[dist] = x1;
    }
}
