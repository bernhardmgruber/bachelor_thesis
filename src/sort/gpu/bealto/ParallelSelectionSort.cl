
__kernel void ParallelSelectionSort(__global const T* in,__global T* out)
{
    int id = get_global_id(0);
    int n = get_global_size(0);

    T element = in[id];

    int pos = 0;
    for (int i=0; i<n; i++)
    {
        T precedingElement = in[i];
        bool smaller = (precedingElement < element) || (precedingElement == element && i < id);
        pos += smaller ? 1 : 0;
    }

    out[pos] = element;
}
