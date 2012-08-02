
#define ORDER(a,b) { bool swap = reverse ^ (a < b); int auxa = a; int auxb = b; a = (swap)?auxb:auxa; b = (swap)?auxa:auxb; }

// N/2 threads
__kernel void ParallelBitonicSortB2(__global int* data, int inc, int dir)
{
    int t = get_global_id(0); // thread index
    int low = t & (inc - 1); // low order bits (below INC)
    int i = (t<<1) - low; // insert 0 at position INC
    bool reverse = ((dir & i) == 0); // asc/desc order
    data += i; // translate to first value

    // Load
    int x0 = data[  0];
    int x1 = data[inc];

    // Sort
    ORDER(x0,x1)

    // Store
    data[0  ] = x0;
    data[inc] = x1;
}
