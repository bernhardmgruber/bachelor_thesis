#ifndef T
#error "T must be defined"
#endif

/**
* From: http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
* Chapter: 39.2.1 A Naive Parallel Scan
*/
__kernel void NaiveScan(__global const T* src, __global T* dest, __local T* shared)
{
    size_t globalId = get_global_id(0);
    size_t k = get_local_id(0);
    size_t n = get_local_size(0);
    
    int pout = 0;
    int pin = 1;  

    // Load input into shared memory.  
    shared[pout * n + k] = src[globalId];
    barrier(CLK_LOCAL_MEM_FENCE); 

    for (uint offset = 1; offset < n; offset <<= 1)  
    {  
        pout = 1 - pout; // swap double buffer indices  
        pin = 1 - pout;  
        if (k >= offset)  
            shared[pout * n + k] = shared[pin * n + k] + shared[pin * n + k - offset];  
        else  
            shared[pout * n + k] = shared[pin * n + k];  
        barrier(CLK_LOCAL_MEM_FENCE);
    }  

    dest[globalId] = shared[pout * n + k]; // write output  
}
