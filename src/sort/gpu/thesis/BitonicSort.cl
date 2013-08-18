__kernel void BitonicSort(__global uint* data, uint inc, uint boxwidth)
{
    uint id = get_global_id(0);
    uint low = id & (inc - 1); // bits below inc
    uint i = (id << 1) - low; // insert 0 at position inc

    bool asc = (i & boxwidth) == 0; // test bit at boxwidth

    data += i;

    uint x0 = data[0];
    uint x1 = data[inc];

    if(asc ^ (x0 < x1))
	{
        data[0  ] = x1;
        data[inc] = x0;
    }
}
