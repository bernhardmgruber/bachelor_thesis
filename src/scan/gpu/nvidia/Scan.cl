#ifndef T
#error "T must be defined"
#endif

/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


//Passed down with -D option on clBuildProgram
//Must be a power of two
//#define WORKGROUP_SIZE 256



////////////////////////////////////////////////////////////////////////////////
// Scan codelets
////////////////////////////////////////////////////////////////////////////////
#if(1)
//Naive inclusive scan: O(N * log2(N)) operations
//Allocate 2 * 'size' local memory, initialize the first half
//with 'size' zeros avoiding if(pos >= offset) condition evaluation
//and saving instructions
inline T scan1Inclusive(T idata, __local T *l_Data, uint size)
{
    uint pos = 2 * get_local_id(0) - (get_local_id(0) & (size - 1));
    l_Data[pos] = 0;
    pos += size;
    l_Data[pos] = idata;

    for(uint offset = 1; offset < size; offset <<= 1)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        T t = l_Data[pos] + l_Data[pos - offset];
        barrier(CLK_LOCAL_MEM_FENCE);
        l_Data[pos] = t;
    }

    return l_Data[pos];
}

inline T scan1Exclusive(T idata, __local T *l_Data, uint size)
{
    return scan1Inclusive(idata, l_Data, size) - idata;
}

#else
#define LOG2_WARP_SIZE 5U
#define      WARP_SIZE (1U << LOG2_WARP_SIZE)

//Almost the same as naive scan1Inclusive but doesn't need barriers
//and works only for size <= WARP_SIZE
inline uint warpScanInclusive(T idata, volatile __local T *l_Data, T size)
{
    uint pos = 2 * get_local_id(0) - (get_local_id(0) & (size - 1));
    l_Data[pos] = 0;
    pos += size;
    l_Data[pos] = idata;

    if(size >=  2) l_Data[pos] += l_Data[pos -  1];
    if(size >=  4) l_Data[pos] += l_Data[pos -  2];
    if(size >=  8) l_Data[pos] += l_Data[pos -  4];
    if(size >= 16) l_Data[pos] += l_Data[pos -  8];
    if(size >= 32) l_Data[pos] += l_Data[pos - 16];

    return l_Data[pos];
}

inline T warpScanExclusive(T idata, __local T *l_Data, uint size)
{
    return warpScanInclusive(idata, l_Data, size) - idata;
}

inline T scan1Inclusive(T idata, __local T *l_Data, uint size)
{
    if(size > WARP_SIZE)
    {
        //Bottom-level inclusive warp scan
        T warpResult = warpScanInclusive(idata, l_Data, WARP_SIZE);

        //Save top elements of each warp for exclusive warp scan
        //sync to wait for warp scans to complete (because l_Data is being overwritten)
        barrier(CLK_LOCAL_MEM_FENCE);
        if( (get_local_id(0) & (WARP_SIZE - 1)) == (WARP_SIZE - 1) )
            l_Data[get_local_id(0) >> LOG2_WARP_SIZE] = warpResult;

        //wait for warp scans to complete
        barrier(CLK_LOCAL_MEM_FENCE);
        if( get_local_id(0) < (WORKGROUP_SIZE / WARP_SIZE) )
        {
            //grab top warp elements
            T val = l_Data[get_local_id(0)];
            //calculate exclsive scan and write back to shared memory
            l_Data[get_local_id(0)] = warpScanExclusive(val, l_Data, size >> LOG2_WARP_SIZE);
        }

        //return updated warp scans with exclusive scan results
        barrier(CLK_LOCAL_MEM_FENCE);
        return warpResult + l_Data[get_local_id(0) >> LOG2_WARP_SIZE];
    }
    else
    {
        return warpScanInclusive(idata, l_Data, size);
    }
}

inline T scan1Exclusive(T idata, __local T *l_Data, uint size)
{
    return scan1Inclusive(idata, l_Data, size) - idata;
}
#endif


//Vector scan: the array to be scanned is stored
//in work-item private memory as uint4
inline T4 scan4Inclusive(T4 data4, __local T *l_Data, uint size)
{
    //Level-0 inclusive scan
    data4.y += data4.x;
    data4.z += data4.y;
    data4.w += data4.z;

    //Level-1 exclusive scan
    T val = scan1Inclusive(data4.w, l_Data, size / 4) - data4.w;

    return (data4 + (T4)val);
}

inline T4 scan4Exclusive(T4 data4, __local T *l_Data, uint size)
{
    return scan4Inclusive(data4, l_Data, size) - data4;
}

////////////////////////////////////////////////////////////////////////////////
// Scan kernels
////////////////////////////////////////////////////////////////////////////////
__kernel __attribute__((reqd_work_group_size(WORKGROUP_SIZE, 1, 1)))
void scanExclusiveLocal1(
    __global T4 *d_Dst,
    __global T4 *d_Src,
    __local T *l_Data,
    uint size
)
{
    //Load data
    T4 idata4 = d_Src[get_global_id(0)];

    //Calculate exclusive scan
    T4 odata4  = scan4Exclusive(idata4, l_Data, size);

    //Write back
    d_Dst[get_global_id(0)] = odata4;
}

//Exclusive scan of top elements of bottom-level scans (4 * THREADBLOCK_SIZE)
__kernel __attribute__((reqd_work_group_size(WORKGROUP_SIZE, 1, 1)))
void scanExclusiveLocal2(
    __global T *d_Buf,
    __global T *d_Dst,
    __global T *d_Src,
    __local T *l_Data,
    uint N,
    uint arrayLength
)
{
    //Load top elements
    //Convert results of bottom-level scan back to inclusive
    //Skip loads and stores for inactive work-items of the work-group with highest index(pos >= N)
    T data = 0;
    if(get_global_id(0) < N)
        data =
            d_Dst[(4 * WORKGROUP_SIZE - 1) + (4 * WORKGROUP_SIZE) * get_global_id(0)] +
            d_Src[(4 * WORKGROUP_SIZE - 1) + (4 * WORKGROUP_SIZE) * get_global_id(0)];

    //Compute
    T odata = scan1Exclusive(data, l_Data, arrayLength);

    //Avoid out-of-bound access
    if(get_global_id(0) < N)
        d_Buf[get_global_id(0)] = odata;
}

//Final step of large-array scan: combine basic inclusive scan with exclusive scan of top elements of input arrays
__kernel __attribute__((reqd_work_group_size(WORKGROUP_SIZE, 1, 1)))
void uniformUpdate(
    __global T4 *d_Data,
    __global T *d_Buf
)
{
    __local T buf[1];

    T4 data4 = d_Data[get_global_id(0)];

    if(get_local_id(0) == 0)
        buf[0] = d_Buf[get_group_id(0)];

    barrier(CLK_LOCAL_MEM_FENCE);
    data4 += (T4)buf[0];
    d_Data[get_global_id(0)] = data4;
}

