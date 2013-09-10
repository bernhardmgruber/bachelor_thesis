#define CONCAT(a, b) a ## b           // concat token a and b
#define CONCAT_EXP(a, b) CONCAT(a, b) // concat token a and b AFTER expansion

inline void order(uint* x, uint a, uint b, bool asc) {
	if(asc ^ (x[a] < x[b])) {
		uint auxa = x[a];
		uint auxb = x[b];
		x[a] = auxb;
		x[b] = auxa;
	} // if
} // order

#define merge1(x, asc)

#define BITONIC_SORT_FUSION(lvl, logLvl, lvlHalf)                                                 \
	inline void CONCAT_EXP(merge, lvl)(uint* x, bool asc) {                                         \
		for (int j = 0; j < lvlHalf; j++)                                                             \
			order(x, j, j + lvlHalf, asc);                                                              \
		CONCAT_EXP(merge, lvlHalf)(x, asc);                                                           \
		CONCAT_EXP(merge, lvlHalf)(x + lvlHalf, asc);                                                 \
	} /* merge */                                                                                   \
                                                                                                  \
	__kernel void CONCAT_EXP(BitonicSortFusion, lvl)(__global uint * data, int inc, int boxwidth) { \
		int id = get_global_id(0);                                                                    \
                                                                                                  \
		inc >>= (logLvl - 1);                                                                         \
		int low = id & (inc - 1);                                                                     \
		int i = ((id - low) << logLvl) + low;                                                         \
		bool asc = ((boxwidth & i) == 0);                                                             \
                                                                                                  \
		data += i;                                                                                    \
                                                                                                  \
		uint x[lvl];                                                                                  \
		for (int k = 0; k < lvl; k++)                                                                 \
			x[k] = data[k * inc];                                                                       \
                                                                                                  \
		CONCAT_EXP(merge, lvl)(x, asc);                                                               \
                                                                                                  \
		for (int k = 0; k < lvl; k++)                                                                 \
			data[k * inc] = x[k];                                                                       \
	} /* BitonicSortFusion */

BITONIC_SORT_FUSION(2, 1, 1);
BITONIC_SORT_FUSION(4, 2, 2);
BITONIC_SORT_FUSION(8, 3, 4);
BITONIC_SORT_FUSION(16, 4, 8);
