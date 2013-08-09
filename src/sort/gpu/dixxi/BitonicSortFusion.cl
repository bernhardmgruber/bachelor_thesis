
#define ORDER(a, b)                \
    do {                           \
    bool swap = asc ^ (a < b); \
    T auxa = a;                    \
    T auxb = b;                    \
    a = swap ? auxb : auxa;        \
    b = swap ? auxa : auxb;        \
    } while(false)


#define ORDERV(x, a, b)                  \
    do {                                 \
    bool swap = asc ^ (x[a] < x[b]); \
    T auxa = x[a];                       \
    T auxb = x[b];                       \
    x[a] = (swap)?auxb:auxa;             \
    x[b] = (swap)?auxa:auxb;             \
    } while(false)

#define B2V(x,a)  do { ORDERV(x, a, a + 1); } while(false)
#define B4V(x,a)  do { for (int ii = 0; ii <  2; ii++) { ORDERV(x, a + ii, a + ii +  2); } B2V (x, a); B2V (x, a +  2); } while(false)
#define B8V(x,a)  do { for (int ii = 0; ii <  4; ii++) { ORDERV(x, a + ii, a + ii +  4); } B4V (x, a); B4V (x, a +  4); } while(false)
#define B16V(x,a) do { for (int ii = 0; ii <  8; ii++) { ORDERV(x, a + ii, a + ii +  8); } B8V (x, a); B8V (x, a +  8); } while(false)
#define B32V(x,a) do { for (int ii = 0; ii < 16; ii++) { ORDERV(x, a + ii, a + ii + 16); } B16V(x, a); B16V(x, a + 16); } while(false)


// N/2 threads
__kernel void BitonicSortFusion2(__global T* data, int inc, int dir)
{
    int t = get_global_id(0); // thread index
    int low = t & (inc - 1); // low order bits (below INC)
    int i = (t<<1) - low; // insert 0 at position INC
    bool asc = ((dir & i) == 0); // asc/desc order
    data += i; // translate to first value

    // Load
    T x[2];
    for (int k=0; k<2; k++)
        x[k] = data[k*inc];

    // Sort
    B2V(x,0);

    // Store
    for (int k=0; k<2; k++)
        data[k*inc] = x[k];
}

// N/4 threads
__kernel void BitonicSortFusion4(__global T * data,int inc,int dir)
{
    inc >>= 1;
    int t = get_global_id(0); // thread index
    int low = t & (inc - 1); // low order bits (below INC)
    int i = ((t - low) << 2) + low; // insert 00 at position INC
    bool asc = ((dir & i) == 0); // asc/desc order
    data += i; // translate to first value

    // Load
    T x[4];
    for (int k=0; k<4; k++)
        x[k] = data[k*inc];

    // Sort
    B4V(x,0);

    // Store
    for (int k=0; k<4; k++)
        data[k*inc] = x[k];
}

// N/8 threads
__kernel void BitonicSortFusion8(__global T * data,int inc,int dir)
{
    inc >>= 2;
    int t = get_global_id(0); // thread index
    int low = t & (inc - 1); // low order bits (below INC)
    int i = ((t - low) << 3) + low; // insert 000 at position INC
    bool asc = ((dir & i) == 0); // asc/desc order
    data += i; // translate to first value

    // Load
    T x[8];
    for (int k=0; k<8; k++)
        x[k] = data[k*inc];

    // Sort
    B8V(x,0);

    // Store
    for (int k=0; k<8; k++)
        data[k*inc] = x[k];
}

// N/16 threads
__kernel void BitonicSortFusion16(__global T * data,int inc,int dir)
{
    inc >>= 3;
    int t = get_global_id(0); // thread index
    int low = t & (inc - 1); // low order bits (below INC)
    int i = ((t - low) << 4) + low; // insert 0000 at position INC
    bool asc = ((dir & i) == 0); // asc/desc order
    data += i; // translate to first value

    // Load
    T x[16];
    for (int k=0; k<16; k++)
        x[k] = data[k*inc];

    // Sort
    B16V(x,0);

    // Store
    for (int k=0; k<16; k++)
        data[k*inc] = x[k];
}

// N/32 threads
__kernel void BitonicSortFusion32(__global T * data,int inc,int dir)
{
    inc >>= 4;
    int t = get_global_id(0); // thread index
    int low = t & (inc - 1); // low order bits (below INC)
    int i = ((t - low) << 5) + low; // insert 00000 at position INC
    bool asc = ((dir & i) == 0); // asc/desc order
    data += i; // translate to first value

    // Load
    T x[32];
    for (int k=0; k<32; k++)
        x[k] = data[k*inc];

    // Sort
    B32V(x,0);

    // Store
    for (int k=0; k<32; k++)
        data[k*inc] = x[k];
}