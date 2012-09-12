#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>
#include <string>

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

#endif // UTILS_H
