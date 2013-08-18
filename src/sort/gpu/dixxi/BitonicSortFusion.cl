#define CONCAT(a, b) a ## b
#define CONCAT_EXPANDED(a, b) CONCAT(a, b)

#define ORDER(x, a, b)                   \
    {                                    \
        bool swap = asc ^ (x[a] < x[b]); \
        if(swap) {                       \
            T auxa = x[a];            \
            T auxb = x[b];            \
            x[a] = auxb;                 \
            x[b] = auxa;                 \
        }                                \   
    }

#define ORDER_2(x,a)  ORDER(x, a, a + 1);
#define ORDER_4(x,a)  { for (int ii = 0; ii <  2; ii++) { ORDER(x, a + ii, a + ii +  2) } ORDER_2 (x, a) ORDER_2 (x, a +  2) }
#define ORDER_8(x,a)  { for (int ii = 0; ii <  4; ii++) { ORDER(x, a + ii, a + ii +  4) } ORDER_4 (x, a) ORDER_4 (x, a +  4) }
#define ORDER_16(x,a) { for (int ii = 0; ii <  8; ii++) { ORDER(x, a + ii, a + ii +  8) } ORDER_8 (x, a) ORDER_8 (x, a +  8) }
#define ORDER_32(x,a) { for (int ii = 0; ii < 16; ii++) { ORDER(x, a + ii, a + ii + 16) } ORDER_16(x, a) ORDER_16(x, a + 16) }

#define BITONIC_SORT_FUSION(lvl, loglvl)                                                               \
    __kernel void CONCAT_EXPANDED(BitonicSortFusion, lvl)(__global T * data, int inc, int boxwidth) \
    {                                                                                                  \
        inc >>= (loglvl - 1);                                                                          \
        int id = get_global_id(0);                                                                     \
        int low = id & (inc - 1);                                                                      \
        int i = ((id - low) << loglvl) + low;                                                          \ 
                                                                                                       \ 
        bool asc = ((boxwidth & i) == 0);                                                              \  
                                                                                                       \
        data += i;                                                                                     \
                                                                                                       \
        T x[lvl];                                                                                   \
        for (int k = 0; k < lvl; k++)                                                                  \
            x[k] = data[k * inc];                                                                      \
                                                                                                       \
        CONCAT_EXPANDED(ORDER_, lvl)(x, 0);                                                            \
                                                                                                       \
        for (int k = 0; k < lvl; k++)                                                                  \
            data[k * inc] = x[k];                                                                      \
    }                                                                                                  

BITONIC_SORT_FUSION(2, 1);
BITONIC_SORT_FUSION(4, 2);
BITONIC_SORT_FUSION(8, 3);
BITONIC_SORT_FUSION(16, 4);
BITONIC_SORT_FUSION(32, 4);