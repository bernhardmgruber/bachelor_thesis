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
uint32_t pow2roundup(uint32_t x);

/**
 * Creates a string representation of the given size (a number of bytes).
 * If size < 1024 it will be expressed in bytes.
 * If size < 1024 * 1024 it will be expressed in kilobytes.
 * ...
 */
string sizeToString(size_t size);

#endif // UTILS_H
