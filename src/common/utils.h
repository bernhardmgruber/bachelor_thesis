#ifndef UTILS_H
#define UTILS_H

#include <CL/cl.h>

#include <stdint.h>
#include <string>
#include <typeinfo>
#include <iterator>
#include <algorithm>
#include <iostream>

using namespace std;

/**
 * Returns the next power of two greater or equal to the given number.
 *
 * From: http://stackoverflow.com/questions/364985/algorithm-for-finding-the-smallest-power-of-two-thats-greater-or-equal-to-a-giv
 */
size_t pow2roundup(size_t x);


/**
 * Rounds the value of x up to the next number which is a multiple of multiple.
 */
size_t roundToMultiple(size_t x, size_t multiple);


/**
 * Creates a string representation of the given size (a number of bytes).
 * If size < 1024 it will be expressed in bytes.
 * If size < 1024 * 1024 it will be expressed in kilobytes.
 * ...
 */
string sizeToString(size_t size);

/**
 * Creates a string representation of the given time.
 *
 * @param time The time to convert in seconds.
 * @return The created string representation in the format of xxh xxm xxs xxxms.
 */
string timeToString(double time);

template <typename T>
string getTypeName()
{
    if(typeid(T) == typeid(int) || typeid(T) == typeid(cl_int))
        return "int";
    if(typeid(T) == typeid(float) || typeid(T) == typeid(cl_float))
        return "float";
    if(typeid(T) == typeid(double) || typeid(T) == typeid(cl_double))
        return "double";
    if(typeid(T) == typeid(unsigned int) || typeid(T) == typeid(cl_uint))
        return "uint";
    return "unknown";
}

template <typename T>
void printArr(T* arr, size_t size)
{
    copy(arr, arr + size, ostream_iterator<T>(cout, ","));
    cout << endl;
}

template <typename T>
void printArr2D(T* arr, size_t size, size_t rowLength)
{
    for(size_t i = 0; i < size; i++)
    {
        cout << arr[i] << ",";
        if((i + 1) % rowLength == 0)
            cout << endl;
    }
    cout << endl;
}

unsigned int ctz(unsigned int);

/**
* Calculates the root of a given power of two (value) that is the largest possible power of two which powered by root does not exceeding value.
*/
unsigned int rootPowerOfTwo(unsigned int value, unsigned int root);

#endif // UTILS_H
