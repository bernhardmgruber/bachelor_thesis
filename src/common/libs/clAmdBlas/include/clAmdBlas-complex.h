/***********************************************************************
**	Copyright (C) 2010 Advanced Micro Devices, Inc. All Rights Reserved.
***********************************************************************/

#ifndef CLAMDBLAS_COMPLEX_H_
#define CLAMDBLAS_COMPLEX H_

#ifdef __cplusplus
extern "C" {
#endif

typedef cl_float2 FloatComplex;
typedef cl_double2 DoubleComplex;

static __inline FloatComplex
floatComplex(float real, float imag)
{
    FloatComplex z;
    z.s[0] = real;
    z.s[1] = imag;
    return z;
}

static __inline DoubleComplex
doubleComplex(double real, double imag)
{
    DoubleComplex z;
    z.s[0] = real;
    z.s[1] = imag;
    return z;
}

#define CREAL(v) ((v).s[0])
#define CIMAG(v) ((v).s[1])

#ifdef __cplusplus
}      /* extern "C" { */
#endif

#endif /* CLAMDBLAS_COMPLEX_H_ */
