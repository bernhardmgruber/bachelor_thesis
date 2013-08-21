#define CONCAT(a, b) a ## b
#define CONCAT_EXPANDED(a, b) CONCAT(a, b)

inline void order(uint* x, uint a, uint b, bool asc) {
	if(asc ^ (x[a] < x[b])) {
		uint auxa = x[a];
		uint auxb = x[b];
		x[a] = auxb;
		x[b] = auxa;
	}
}

#define merge1(x, asc)

#define BITONIC_SORT_FUSION(lvl, logLvl, lvlHalf)                                                      \
	inline void CONCAT_EXPANDED(merge, lvl)(uint* x, bool asc) {                                         \
		for (int j = 0; j < lvlHalf; j++)                                                                  \
			order(x, j, j + lvlHalf, asc);                                                                   \
		CONCAT_EXPANDED(merge, lvlHalf)(x, asc);                                                           \
		CONCAT_EXPANDED(merge, lvlHalf)(x + lvlHalf, asc);                                                 \
	}                                                                                                    \
                                                                                                       \
	__kernel void CONCAT_EXPANDED(BitonicSortFusion, lvl)(__global uint * data, int inc, int boxwidth) { \
		int id = get_global_id(0);                                                                         \
                                                                                                       \
		inc >>= (logLvl - 1);                                                                              \
		int low = id & (inc - 1);                                                                          \
		int i = ((id - low) << logLvl) + low;                                                              \
		bool asc = ((boxwidth & i) == 0);                                                                  \
                                                                                                       \
		data += i;                                                                                         \
                                                                                                       \
		uint x[lvl];                                                                                       \
		for (int k = 0; k < lvl; k++)                                                                      \
			x[k] = data[k * inc];                                                                            \
                                                                                                       \
		CONCAT_EXPANDED(merge, lvl)(x, asc);                                                               \
                                                                                                       \
		for (int k = 0; k < lvl; k++)                                                                      \
			data[k * inc] = x[k];                                                                            \
	}

BITONIC_SORT_FUSION(2, 1, 1);
BITONIC_SORT_FUSION(4, 2, 2);
BITONIC_SORT_FUSION(8, 3, 4);
BITONIC_SORT_FUSION(16, 4, 8);
