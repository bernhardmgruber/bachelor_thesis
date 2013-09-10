size_t roundToPowerOfTwo(size_t x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    if (sizeof(size_t) >= 8)
        x |= x >> 32;

    return x + 1;
} // roundToPowerOfTwo

size_t roundToMultiple(size_t x, size_t multiple) {
    if (x % multiple == 0)
        return x;
    else
        return (x / multiple + 1) * multiple;
} // roundToMultiple
