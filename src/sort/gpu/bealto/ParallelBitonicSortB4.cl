
#define ORDER(a,b) { bool swap = reverse ^ (a < b); T auxa = a; T auxb = b; a = (swap)?auxb:auxa; b = (swap)?auxa:auxb; }

// N/2 threads
__kernel void ParallelBitonicSortB2(__global T* data, int inc, int dir)
{
    int t = get_global_id(0); // thread index
    int low = t & (inc - 1); // low order bits (below INC)
    int i = (t<<1) - low; // insert 0 at position INC
    bool reverse = ((dir & i) == 0); // asc/desc order
    data += i; // translate to first value

    // Load
    T x0 = data[  0];
    T x1 = data[inc];

    // Sort
    ORDER(x0,x1)

    // Store
    data[0  ] = x0;
    data[inc] = x1;
}

// N/4 threads
__kernel void ParallelBitonicSortB4(__global int * data,int inc,int dir)
{
    inc >>= 1;
    int t = get_global_id(0); // thread index
    int low = t & (inc - 1); // low order bits (below INC)
    int i = ((t - low) << 2) + low; // insert 00 at position INC
    bool reverse = ((dir & i) == 0); // asc/desc order
    data += i; // translate to first value

    // Load
    T x0 = data[    0];
    T x1 = data[  inc];
    T x2 = data[2*inc];
    T x3 = data[3*inc];

    // Sort
    ORDER(x0,x2)
    ORDER(x1,x3)
    ORDER(x0,x1)
    ORDER(x2,x3)

    // Store
    data[    0] = x0;
    data[  inc] = x1;
    data[2*inc] = x2;
    data[3*inc] = x3;
}
