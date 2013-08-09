
// N / 2 threads
__kernel void BitonicSort(__global T* data, uint inc, uint boxwidth)
{
    uint id = get_global_id(0); // thread index
    uint low = id & (inc - 1); // low order bits (below dist)
    uint i = (id << 1) - low; // insert 0 at position dist

    bool asc = ((boxwidth & i) == 0); // test if the bit at boxwidth is set, determines asc/desc order

    data += i; // translate to first value

    T x0 = data[0];
    T x1 = data[inc];

    bool swap = asc ^ (x0 < x1);
    if(swap) {
        data[0  ] = x1;
        data[inc] = x0;
    }
}
