
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
    int x0 = data[    0];
    int x1 = data[  inc];
    int x2 = data[2*inc];
    int x3 = data[3*inc];

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

#define ORDERV(x,a,b) { bool swap = reverse ^ (x[a] < x[b]); \
      int auxa = x[a]; int auxb = x[b]; \
      x[a] = (swap)?auxb:auxa; x[b] = (swap)?auxa:auxb; }
#define B2V(x,a) { ORDERV(x,a,a+1) }
#define B4V(x,a) { for (int i4=0;i4<2;i4++) { ORDERV(x,a+i4,a+i4+2) } B2V(x,a) B2V(x,a+2) }
#define B8V(x,a) { for (int i8=0;i8<4;i8++) { ORDERV(x,a+i8,a+i8+4) } B4V(x,a) B4V(x,a+4) }
#define B16V(x,a) { for (int i16=0;i16<8;i16++) { ORDERV(x,a+i16,a+i16+8) } B8V(x,a) B8V(x,a+8) }

// N/8 threads
__kernel void ParallelBitonicSortB8(__global int * data,int inc,int dir)
{
  inc >>= 2;
  int t = get_global_id(0); // thread index
  int low = t & (inc - 1); // low order bits (below INC)
  int i = ((t - low) << 3) + low; // insert 000 at position INC
  bool reverse = ((dir & i) == 0); // asc/desc order
  data += i; // translate to first value

  // Load
  int x[8];
  for (int k=0;k<8;k++) x[k] = data[k*inc];

  // Sort
  B8V(x,0)

  // Store
  for (int k=0;k<8;k++) data[k*inc] = x[k];
}
