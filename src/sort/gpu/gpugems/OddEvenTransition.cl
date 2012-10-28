
__kernel void OddEvenTransition(__global T * buffer)
{
    size_t id = get_global_id(0);

    // EVEN
    T a = buffer[id * 2 + 0];
    T b = buffer[id * 2 + 1];

    if(a > b)
    {
        buffer[id * 2 + 0] = b;
        buffer[id * 2 + 1] = a;
    }

    // ODD
    if(id != get_global_size(0) - 1)
    {
        a = buffer[id * 2 + 1];
        b = buffer[id * 2 + 2];

        if(a > b)
        {
            buffer[id * 2 + 1] = b;
            buffer[id * 2 + 2] = a;
        }
    }
}
