
/***********************************************************************
** Copyright (C) 2010,2011 Advanced Micro Devices, Inc. All Rights Reserved.
***********************************************************************/

#ifndef CLAMDBLAS_H_
#define CLAMDBLAS_H_

/**
 * @mainpage OpenCL BLAS-3
 *
 * This is an implementation of
 * <A HREF="http://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms">
 * Basic Linear Algebra Subprograms</A>, levels 2 and 3 using
 * <A HREF="http://www.khronos.org/opencl/">OpenCL</A> and optimized for
 * the AMD GPU hardware.
 */

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <clAmdBlas-complex.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup OVERVIEW Overview
 *
 * This library provides an implementation of the Basic Linear Algebra Subprograms levels 2 and 3,
 * using OpenCL and optimized for AMD GPU hardware. It provides
 * BLAS-2 functions GEMV and SYMV, and BLAS-3 functions
 * GEMM, TRMM, TRSM, SYRK, and SYR2K. GEMV, SYMV, SYRK and SYR2K support
 * single and double precision data types and all other functions support single,
 * double, complex single, and complex double precision data types.
 *
 * This library’s primary goal is to assist the end user to enqueue OpenCL
 * kernels to process BLAS functions in an OpenCL-efficient manner, while
 * keeping interfaces familiar to users who know how to use BLAS. All
 * functions accept matrices through buffer objects.
 *
 * @section deprecated
 * This library provided support for the creation of scratch images to achieve better performance
 * on older <a href="http://developer.amd.com/gpu/AMDAPPSDK/Pages/default.aspx">AMD APP SDK's</a>. 
 * However, memory buffers now give the same performance as buffers objects in the current SDK's. 
 * Scratch image buffers are being deprecated and users are advised not to use scratch images in
 * new applications.
 */

/**
 * @defgroup TYPES clAmdBlas types
 */
/*@{*/


/** Shows how matrices are placed in memory. */
typedef enum clAmdBlasOrder {
    clAmdBlasRowMajor,           /**< Every row is placed sequentially */
    clAmdBlasColumnMajor         /**< Every column is placed sequentially */
} clAmdBlasOrder;

/** Used to specify whether the matrix is to be transposed or not. */
typedef enum clAmdBlasTranspose {
    clAmdBlasNoTrans,           /**< Operate with the matrix. */
    clAmdBlasTrans,             /**< Operate with the transpose of the matrix. */
    clAmdBlasConjTrans          /**< Operate with the conjugate transpose of
                                     the matrix. */
} clAmdBlasTranspose;

/** Used by the Hermitian, symmetric and triangular matrix
 * routines to specify whether the upper or lower triangle is being referenced.
 */
typedef enum clAmdBlasUplo {
    clAmdBlasUpper,               /**< Upper triangle. */
    clAmdBlasLower                /**< Lower triangle. */
} clAmdBlasUplo;

/** It is used by the triangular matrix routines to specify whether the
 * matrix is unit triangular.
 */
typedef enum clAmdBlasDiag {
    clAmdBlasUnit,               /**< Unit triangular. */
    clAmdBlasNonUnit             /**< Non-unit triangular. */
} clAmdBlasDiag;

/** Indicates the side matrix A is located relative to matrix B during multiplication. */
typedef enum clAmdBlasSide {
    clAmdBlasLeft,        /**< Multiply general matrix by symmetric,
                               Hermitian or triangular matrix on the left. */
    clAmdBlasRight        /**< Multiply general matrix by symmetric,
                               Hermitian or triangular matrix on the right. */
} clAmdBlasSide;

/**
 *   @brief clAmdBlas error codes definition, incorporating OpenCL error
 *   definitions.
 *
 *   This enumeration is a subset of the OpenCL error codes extended with some
 *   additional extra codes.  For example, CL_OUT_OF_HOST_MEMORY, which is
 *   defined in cl.h is aliased as clAmdBlasOutOfHostMemory.
 */
typedef enum clAmdBlasStatus {
    clAmdBlasSuccess                         = CL_SUCCESS,
    clAmdBlasInvalidValue                    = CL_INVALID_VALUE,
    clAmdBlasInvalidCommandQueue             = CL_INVALID_COMMAND_QUEUE,
    clAmdBlasInvalidContext                  = CL_INVALID_CONTEXT,
    clAmdBlasInvalidMemObject                = CL_INVALID_MEM_OBJECT,
    clAmdBlasInvalidDevice                   = CL_INVALID_DEVICE,
    clAmdBlasInvalidEventWaitList            = CL_INVALID_EVENT_WAIT_LIST,
    clAmdBlasOutOfResources                  = CL_OUT_OF_RESOURCES,
    clAmdBlasOutOfHostMemory                 = CL_OUT_OF_HOST_MEMORY,
    clAmdBlasInvalidOperation                = CL_INVALID_OPERATION,
    clAmdBlasCompilerNotAvailable            = CL_COMPILER_NOT_AVAILABLE,
    clAmdBlasBuildProgramFailure             = CL_BUILD_PROGRAM_FAILURE,
    /* Extended error codes */
    clAmdBlasNotImplemented         = -1024, /**< Functionality is not implemented */
    clAmdBlasNotInitialized,                 /**< clAmdBlas library is not initialized yet */
    clAmdBlasInvalidMatA,                    /**< Matrix A is not a valid memory object */
    clAmdBlasInvalidMatB,                    /**< Matrix B is not a valid memory object */
    clAmdBlasInvalidMatC,                    /**< Matrix C is not a valid memory object */
    clAmdBlasInvalidVecX,                    /**< Vector X is not a valid memory object */
    clAmdBlasInvalidVecY,                    /**< Vector Y is not a valid memory object */
    clAmdBlasInvalidDim,                     /**< An input dimension (M,N,K) is invalid */
    clAmdBlasInvalidLeadDimA,                /**< Leading dimension A must not be less than the size of the first dimension */
    clAmdBlasInvalidLeadDimB,                /**< Leading dimension B must not be less than the size of the second dimension */
    clAmdBlasInvalidLeadDimC,                /**< Leading dimension C must not be less than the size of the third dimension */
    clAmdBlasInvalidIncX,                    /**< The increment for a vector X must not be 0 */
    clAmdBlasInvalidIncY,                    /**< The increment for a vector Y must not be 0 */
    clAmdBlasInsufficientMemMatA,            /**< The memory object for Matrix A is too small */
    clAmdBlasInsufficientMemMatB,            /**< The memory object for Matrix B is too small */
    clAmdBlasInsufficientMemMatC,            /**< The memory object for Matrix C is too small */
    clAmdBlasInsufficientMemVecX,            /**< The memory object for Vector X is too small */
    clAmdBlasInsufficientMemVecY             /**< The memory object for Vector Y is too small */
} clAmdBlasStatus;


/*@}*/

/**
 * @defgroup VERSION Version information
 */
/*@{*/

/**
 * @brief Get the clAmdBlas library version info.
 *
 * @param[out] major        Location to store library's major version.
 * @param[out] minor        Location to store library's minor version.
 * @param[out] patch        Location to store library's patch version.
 *
 * @returns always \b clAmdBlasSuccess.
 *
 * @ingroup VERSION
 */
clAmdBlasStatus
clAmdBlasGetVersion(cl_uint* major, cl_uint* minor, cl_uint* patch);

/*@}*/

/**
 * @defgroup INIT Initialize library
 */
/*@{*/

/**
 * @brief Initialize the clAmdBlas library.
 *
 * Must be called before any other clAmdBlas API function is invoked.
 * @note This function is not thread-safe.
 *
 * @return
 *   - \b clAmdBlasSucces on success;
 *   - \b clAmdBlasOutOfHostMemory if there is not enough of memory to allocate
 *     library's internal structures;
 *   - \b clAmdBlasOutOfResources in case of requested resources scarcity.
 *
 * @ingroup INIT
 */
clAmdBlasStatus
clAmdBlasSetup(void);

/**
 * @brief Finalize the usage of the clAmdBlas library.
 *
 * Frees all memory allocated for different computational kernel and other
 * internal data.
 * @note This function is not thread-safe.
 *
 * @ingroup INIT
 */
void
clAmdBlasTeardown(void);

/*@}*/

/**
 * @defgroup MISC Miscellaneous
 */
/*@{*/

/**
 * @deprecated
 * @brief Create scratch image.
 *
 * Images created with this function can be used by the library to switch from
 * a buffer-based to an image-based implementation. This can increase
 * performance up to 2 or 3 times over buffer-objects-based ones on same systems.
 * To leverage the GEMM and TRMM kernels, it is necessary to create two images.
 *
 * The following description provides bounds for the width and height arguments
 * for functions that can use scratch images.
 *
 * Let \c type be the data type of the function in question.
 *
 * Let <tt>fl4RelSize(type) = sizeof(cl_float4) / sizeof(type)</tt>.
 *
 * Let \c width1 and \c width2 be the respective values of the width argument
 * passed into the function for the two images needed to activate the image-based
 * algorithm. Similarly, let \c height1 and \c height2 be the values for the
 * height argument.
 *
 * Let <tt>div_up(x,y) = (x + y – 1) / y</tt>.
 *
 * Let <tt>round_up(x,y) = div_up(x,y) * y</tt>.
 *
 * Let <tt>round_down(x,y) = (x / y) * y</tt>.
 *
 * Then:
 *
 * For \b xGEMM there should be 2 images satisfying the following requirements:
 *   - <tt>width1 >= round_up(K, 64) / fl4RelSize(type)</tt>,
 *   - <tt>width2 >= round_up(K, 64) / fl4RelSize(type)</tt>,
 *   - <tt>height >= 64M</tt>,
 *
 * for any transA, transB, and order.
 *
 * For \b xTRMM:
 *   - <tt>width1 >= round_up(T, 64) / fl4RelSize(type)</tt>,
 *   - <tt>width2 >= round_up(N, 64) / fl4RelSize(type)</tt>,
 *   - <tt>height >= 64</tt>,
 *
 * for any transA, transB and order, where
 *   - \c T = M, for \c side = clAmdBlasLeft, and
 *   - \c T = N, for \c side = clAmdBlasRight.
 *
 * For \b xTRSM:
 *   - <tt>round_down(width, 32) * round_down(height, 32) * fl4RelSize(type) >= 1/2 * (round_up(T, 32)^2 + div_up(T, 32) * 32^2)</tt>
 *
 * for any transA, transB and order, where
 *   - \c T = M, for \c side = clAmdBlasLeft, and
 *   - \c T = N, for \c side = clAmdBlasRight.
 *
 * A call to clAmdAddScratchImage with arguments \c width and \c height reserves
 * approximately <tt>width * height * 16</tt> bytes of device memory.
 *
 * @return A created image identifier.
 *
 * @ingroup MISC
 */
cl_ulong
clAmdBlasAddScratchImage(
    cl_context context,
    size_t width,
    size_t height,
    clAmdBlasStatus *status);

/**
 * @deprecated
 * @brief Release scratch image.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if an invalid image identified is passed.
 *
 * @ingroup MISC
 */
clAmdBlasStatus
clAmdBlasRemoveScratchImage(
    cl_ulong imageID);

/*@}*/

/**
 * @defgroup BLAS2 BLAS-2 functions
 *
 * The Level 2 Basic Linear Algebra Subprograms are functions that perform
 * matrix-vector operations.
 */
/*@{*/
/*@}*/


/**
 * @defgroup GEMV GEMV  - GEneral Matrix-Vector multiplication
 * @ingroup BLAS2
 */
/*@{*/

/**
 * @brief Matrix-vector product with a general rectangular matrix and
 * float elements.
 *
 * Matrix-vector products:
 *   - \f$ y \leftarrow \alpha A x + \beta y \f$
 *   - \f$ y \leftarrow \alpha A^T x + \beta y \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b M when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b N when the
 *                      parameter is set to \b clAmdBlasColumnMajor.
 * @param[in] x         Buffer object storing vector \b x.
 * @param[in] offx      Offset of first element of vector \b x in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b x. Must not be zero.
 * @param[in] beta      The factor of the vector \b y.
 * @param[out] y        Buffer object storing the vector \b y.
 * @param[in] offy      Offset of first element of vector \b y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b y. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasSgemvEx() instead.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - either \b M or \b N is zero, or
 *     - either \b incx or \b incy is zero, or
 *     - any of the leading dimensions is invalid;
 *     - the matrix size or the vector sizes along with the increments lead to
 *       accessing outside of any of the buffers;
 *   - \b clAmdBlasInvalidMemObject if either \b A, \b x, or \b y object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup GEMV
 */
clAmdBlasStatus
clAmdBlasSgemv(
    clAmdBlasOrder order,
    clAmdBlasTranspose transA,
    size_t M,
    size_t N,
    cl_float alpha,
    const cl_mem A,
    size_t lda,
    const cl_mem x,
    size_t offx,
    int incx,
    cl_float beta,
    cl_mem y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @example example_sgemv.c
 * Example of how to use the @ref clAmdBlasSgemv function.
 */

/**
 * @brief Matrix-vector product with a general rectangular matrix and
 * double elements.
 *
 * Matrix-vector products:
 *   - \f$ y \leftarrow \alpha A x + \beta y \f$
 *   - \f$ y \leftarrow \alpha A^T x + \beta y \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. For a detailed description,
 *                      see clAmdBlasSgemv().
 * @param[in] x         Buffer object storing vector \b x.
 * @param[in] offx      Offset of first element of vector \b x in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b x. It cannot be zero.
 * @param[in] beta      The factor of the vector \b y.
 * @param[out] y        Buffer object storing the vector \b y.
 * @param[in] offy      Offset of first element of vector \b y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasDgemvEx() instead.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support the
 *     floating point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasSgemv() function otherwise.
 *
 * @ingroup GEMV
 */
clAmdBlasStatus
clAmdBlasDgemv(
    clAmdBlasOrder order,
    clAmdBlasTranspose transA,
    size_t M,
    size_t N,
    cl_double alpha,
    const cl_mem A,
    size_t lda,
    const cl_mem x,
    size_t offx,
    int incx,
    cl_double beta,
    cl_mem y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Matrix-vector product with a general rectangular matrix and
 * float complex elements.
 *
 * Matrix-vector products:
 *   - \f$ y \leftarrow \alpha A x + \beta y \f$
 *   - \f$ y \leftarrow \alpha A^T x + \beta y \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. For a detailed description,
 *                      see clAmdBlasSgemv().
 * @param[in] x         Buffer object storing vector \b x.
 * @param[in] offx      Offset of first element of vector \b x in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b x. It cannot be zero.
 * @param[in] beta      The factor of the vector \b y.
 * @param[out] y        Buffer object storing the vector \b y.
 * @param[in] offy      Offset of first element of vector \b y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasCgemvEx() instead.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - the same error codes as the clAmdBlasSgemv() function otherwise.
 *
 * @ingroup GEMV
 */
clAmdBlasStatus
clAmdBlasCgemv(
    clAmdBlasOrder order,
    clAmdBlasTranspose transA,
    size_t M,
    size_t N,
    FloatComplex alpha,
    const cl_mem A,
    size_t lda,
    const cl_mem x,
    size_t offx,
    int incx,
    FloatComplex beta,
    cl_mem y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Matrix-vector product with a general rectangular matrix and
 * double complex elements.
 *
 * Matrix-vector products:
 *   - \f$ y \leftarrow \alpha A x + \beta y \f$
 *   - \f$ y \leftarrow \alpha A^T x + \beta y \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. For a detailed description,
 *                      see clAmdBlasSgemv().
 * @param[in] x         Buffer object storing vector \b x.
 * @param[in] offx      Offset of first element of vector \b x in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b x. It cannot be zero.
 * @param[in] beta      The factor of the vector \b y.
 * @param[out] y        Buffer object storing the vector \b y.
 * @param[in] offy      Offset of first element of vector \b y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasZgemvEx() instead.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support the
 *     floating point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasSgemv() function otherwise.
 *
 * @ingroup GEMV
 */
clAmdBlasStatus
clAmdBlasZgemv(
    clAmdBlasOrder order,
    clAmdBlasTranspose transA,
    size_t M,
    size_t N,
    DoubleComplex alpha,
    const cl_mem A,
    size_t lda,
    const cl_mem x,
    size_t offx,
    int incx,
    DoubleComplex beta,
    cl_mem y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Matrix-vector product with a general rectangular matrix and
 *        float elements. Extended version.
 *
 * Matrix-vector products:
 *   - \f$ y \leftarrow \alpha A x + \beta y \f$
 *   - \f$ y \leftarrow \alpha A^T x + \beta y \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in
 *                      the buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b M when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b N when the
 *                      parameter is set to \b clAmdBlasColumnMajor.
 * @param[in] x         Buffer object storing vector \b x.
 * @param[in] offx      Offset of first element of vector \b x in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b x. It cannot be zero.
 * @param[in] beta      The factor of the vector \b y.
 * @param[out] y        Buffer object storing the vector \b y.
 * @param[in] offy      Offset of first element of vector \b y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidValue if \b offA exceeds the size of \b A buffer
 *     object;
 *   - the same error codes as the clAmdBlasSgemv() function otherwise.
 *
 * @ingroup GEMV
 */
clAmdBlasStatus
clAmdBlasSgemvEx(
    clAmdBlasOrder order,
    clAmdBlasTranspose transA,
    size_t M,
    size_t N,
    cl_float alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem x,
    size_t offx,
    int incx,
    cl_float beta,
    cl_mem y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @example example_sgemv.c
 * This is an example of how to use the @ref clAmdBlasSgemvEx function.
 */

/**
 * @brief Matrix-vector product with a general rectangular matrix and
 *        double elements. Extended version.
 *
 * Matrix-vector products:
 *   - \f$ y \leftarrow \alpha A x + \beta y \f$
 *   - \f$ y \leftarrow \alpha A^T x + \beta y \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of \b A in the buffer
 *                      object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. For a detailed description,
 *                      see clAmdBlasSgemv().
 * @param[in] x         Buffer object storing vector \b x.
 * @param[in] offx      Offset of first element of vector \b x in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b x. It cannot be zero.
 * @param[in] beta      The factor of the vector \b y.
 * @param[out] y        Buffer object storing the vector \b y.
 * @param[in] offy      Offset of first element of vector \b y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support the
 *     floating point arithmetic with double precision;
 *   - \b clAmdBlasInvalidValue if \b offA exceeds the size of \b A buffer
 *     object;
 *   - the same error codes as the clAmdBlasSgemv() function otherwise.
 *
 * @ingroup GEMV
 */
clAmdBlasStatus
clAmdBlasDgemvEx(
    clAmdBlasOrder order,
    clAmdBlasTranspose transA,
    size_t M,
    size_t N,
    cl_double alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem x,
    size_t offx,
    int incx,
    cl_double beta,
    cl_mem y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Matrix-vector product with a general rectangular matrix and
 *        float complex elements. Extended version.
 *
 * Matrix-vector products:
 *   - \f$ y \leftarrow \alpha A x + \beta y \f$
 *   - \f$ y \leftarrow \alpha A^T x + \beta y \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in
 *                      the buffer object. Counted in elements
 * @param[in] lda       Leading dimension of matrix \b A. For a detailed description,
 *                      see clAmdBlasSgemv().
 * @param[in] x         Buffer object storing vector \b x.
 * @param[in] offx      Offset of first element of vector \b x in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b x. It cannot be zero.
 * @param[in] beta      The factor of the vector \b y.
 * @param[out] y        Buffer object storing the vector \b y.
 * @param[in] offy      Offset of first element of vector \b y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidValue if \b offA exceeds the size of \b A buffer
 *     object;
 *   - the same error codes as the clAmdBlasSgemv() function otherwise.
 *
 * @ingroup GEMV
 */
clAmdBlasStatus
clAmdBlasCgemvEx(
    clAmdBlasOrder order,
    clAmdBlasTranspose transA,
    size_t M,
    size_t N,
    FloatComplex alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem x,
    size_t offx,
    int incx,
    FloatComplex beta,
    cl_mem y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Matrix-vector product with a general rectangular matrix and
 *        double complex elements. Extended version.
 *
 * Matrix-vector products:
 *   - \f$ y \leftarrow \alpha A x + \beta y \f$
 *   - \f$ y \leftarrow \alpha A^T x + \beta y \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in
 *                      the buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. For a detailed description,
 *                      see clAmdBlasSgemv().
 * @param[in] x         Buffer object storing vector \b x.
 * @param[in] offx      Offset of first element of vector \b x in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b x. It cannot be zero.
 * @param[in] beta      The factor of the vector \b y.
 * @param[out] y        Buffer object storing the vector \b y.
 * @param[in] offy      Offset of first element of vector \b y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support the
 *     floating point arithmetic with double precision;
 *   - \b clAmdBlasInvalidValue if \b offA exceeds the size of \b A buffer
 *     object;
 *   - the same error codes as the clAmdBlasSgemv() function otherwise.
 *
 * @ingroup GEMV
 */
clAmdBlasStatus
clAmdBlasZgemvEx(
    clAmdBlasOrder order,
    clAmdBlasTranspose transA,
    size_t M,
    size_t N,
    DoubleComplex alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem x,
    size_t offx,
    int incx,
    DoubleComplex beta,
    cl_mem y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/*@}*/

/**
 * @defgroup SYMV SYMV  - SYmmetric Matrix-Vector multiplication
 * @ingroup BLAS2
 */

/*@{*/

/**
 * @brief Matrix-vector product with a symmetric matrix and float elements.
 *
 * Matrix-vector products:
 * - \f$ y \leftarrow \alpha A x + \beta y \f$
 *
 * @param[in] order     Row/columns order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of rows and columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot less
 *                      than \b N.
 * @param[in] x         Buffer object storing vector \b x.
 * @param[in] offx      Offset of first element of vector \b x in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of vector \b x. It cannot be zero.
 * @param[in] beta      The factor of vector \b y.
 * @param[out] y        Buffer object storing vector \b y.
 * @param[in] offy      Offset of first element of vector \b y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of vector \b y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasSsymvEx() instead.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - either \b incx or \b incy is zero, or
 *     - any of the leading dimensions is invalid;
 *     - the matrix sizes or the vector sizes along with the increments lead to
 *       accessing outsize of any of the buffers;
 *   - \b clAmdBlasInvalidMemObject if either \b A, \b x, or \b y object is
 *     invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs to
 *     was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup SYMV
 */
clAmdBlasStatus
clAmdBlasSsymv(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    size_t N,
    cl_float alpha,
    const cl_mem A,
    size_t lda,
    const cl_mem x,
    size_t offx,
    int incx,
    cl_float beta,
    cl_mem y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @example example_ssymv.c
 * This is an example of how to use the @ref clAmdBlasSsymv function.
 */

/**
 * @brief Matrix-vector product with a symmetric matrix and double elements.
 *
 * Matrix-vector products:
 * - \f$ y \leftarrow \alpha A x + \beta y \f$
 *
 * @param[in] order     Row/columns order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of rows and columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot less
 *                      than \b N.
 * @param[in] x         Buffer object storing vector \b x.
 * @param[in] offx      Offset of first element of vector \b x in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of vector \b x. It cannot be zero.
 * @param[in] beta      The factor of vector \b y.
 * @param[out] y        Buffer object storing vector \b y.
 * @param[in] offy      Offset of first element of vector \b y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of vector \b y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasDsymvEx() instead.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasSsymv() function otherwise.
 *
 * @ingroup SYMV
 */
clAmdBlasStatus
clAmdBlasDsymv(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    size_t N,
    cl_double alpha,
    const cl_mem A,
    size_t lda,
    const cl_mem x,
    size_t offx,
    int incx,
    cl_double beta,
    cl_mem y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Matrix-vector product with a symmetric matrix and float elements.
 *        Extended version.
 *
 * Matrix-vector products:
 * - \f$ y \leftarrow \alpha A x + \beta y \f$
 *
 * @param[in] order     Row/columns order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of rows and columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in
 *                      the buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot less
 *                      than \b N.
 * @param[in] x         Buffer object storing vector \b x.
 * @param[in] offx      Offset of first element of vector \b x in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of vector \b x. It cannot be zero.
 * @param[in] beta      The factor of vector \b y.
 * @param[out] y        Buffer object storing vector \b y.
 * @param[in] offy      Offset of first element of vector \b y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of vector \b y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidValue if \b offA exceeds the size of \b A buffer
 *     object;
 *   - the same error codes as the clAmdBlasSgemv() function otherwise.
 *
 * @ingroup SYMV
 */
clAmdBlasStatus
clAmdBlasSsymvEx(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    size_t N,
    cl_float alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem x,
    size_t offx,
    int incx,
    cl_float beta,
    cl_mem y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @example example_ssymv.c
 * This is an example of how to use the @ref clAmdBlasSsymv function.
 */

/**
 * @brief Matrix-vector product with a symmetric matrix and double elements.
 *        Extended version.
 *
 * Matrix-vector products:
 * - \f$ y \leftarrow \alpha A x + \beta y \f$
 *
 * @param[in] order     Row/columns order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of rows and columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in
 *                      the buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot less
 *                      than \b N.
 * @param[in] x         Buffer object storing vector \b x.
 * @param[in] offx      Offset of first element of vector \b x in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of vector \b x. It cannot be zero.
 * @param[in] beta      The factor of vector \b y.
 * @param[out] y        Buffer object storing vector \b y.
 * @param[in] offy      Offset of first element of vector \b y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of vector \b y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - \b clAmdBlasInvalidValue if \b offA exceeds the size of \b A buffer
 *     object;
 *   - the same error codes as the clAmdBlasSsymv() function otherwise.
 *
 * @ingroup SYMV
 */
clAmdBlasStatus
clAmdBlasDsymvEx(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    size_t N,
    cl_double alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem x,
    size_t offx,
    int incx,
    cl_double beta,
    cl_mem y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/*@}*/


/**
 * @defgroup HEMV HEMV  - HErmitian Matrix-Vector multiplication
 * @ingroup BLAS2
 */
/*@{*/

/**
 * @brief Matrix-vector product with a hermitian matrix and float-complex elements.
 *
 * Matrix-vector products:
 * - \f$ Y \leftarrow \alpha A X + \beta Y \f$
 *
 * @param[in] order     Row/columns order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of rows and columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offa		Offset in number of elements for first element in matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot less
 *                      than \b N.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of vector \b X. It cannot be zero.
 * @param[in] beta      The factor of vector \b Y.
 * @param[out] Y        Buffer object storing vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of vector \b Y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - either \b incx or \b incy is zero, or
 *     - any of the leading dimensions is invalid;
 *     - the matrix sizes or the vector sizes along with the increments lead to
 *       accessing outsize of any of the buffers;
 *   - \b clAmdBlasInvalidMemObject if either \b A, \b X, or \b Y object is
 *     invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs to
 *     was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup HEMV
 */
clAmdBlasStatus
clAmdBlasChemv(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    size_t N,
    FloatComplex alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem X,
    size_t offx,
    int incx,
    FloatComplex beta,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Matrix-vector product with a hermitian matrix and double-complex elements.
 *
 * Matrix-vector products:
 * - \f$ Y \leftarrow \alpha A X + \beta Y \f$
 *
 * @param[in] order     Row/columns order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of rows and columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offa		Offset in number of elements for first element in matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot less
 *                      than \b N.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of vector \b X. It cannot be zero.
 * @param[in] beta      The factor of vector \b Y.
 * @param[out] Y        Buffer object storing vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of vector \b Y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasChemv() function otherwise.
 *
 * @ingroup HEMV
 */
clAmdBlasStatus
clAmdBlasZhemv(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    size_t N,
    DoubleComplex alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem X,
    size_t offx,
    int incx,
    DoubleComplex beta,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/**
 * @example example_zhemv.cpp
 * Example of how to use the @ref clAmdBlasZhemv function.
 */
/*@}*/



/**
 * @defgroup TRMV TRMV  - TRiangular Matrix Vector multiply
 * @ingroup BLAS2
 */
/*@{*/

/**
 * @brief Matrix-vector product with a triangular matrix and
 * float elements.
 *
 * Matrix-vector products:
 *   - \f$ X \leftarrow  A X \f$
 *   - \f$ X \leftarrow  A^T X \f$
 *
 * @param[in] order				Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b A is to be transposed.
 * @param[in] diag				Specify whether matrix \b A is unit triangular.
 * @param[in] N					Number of rows/columns in matrix \b A.
 * @param[in] A					Buffer object storing matrix \b A.
 * @param[in] offa				Offset in number of elements for first element in matrix \b A.
 * @param[in] lda				Leading dimension of matrix \b A. It cannot be less
 *								than \b N
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff		Temporary cl_mem scratch buffer object which can hold a
 *								minimum of (1 + (N-1)*abs(incx)) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - either \b N or \b incx is zero, or
 *     - the leading dimension is invalid;
 *   - \b clAmdBlasInvalidMemObject if either \b A or \b X object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup TRMV
 */
clAmdBlasStatus 
clAmdBlasStrmv(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose trans,
    clAmdBlasDiag diag,
    size_t N,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
	cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @example example_strmv.c
 * Example of how to use the @ref clAmdBlasStrmv function.
 */

/**
 * @brief Matrix-vector product with a triangular matrix and
 * double elements.
 *
 * Matrix-vector products:
 *   - \f$ X \leftarrow  A X \f$
 *   - \f$ X \leftarrow  A^T X \f$
 *
 * @param[in] order				Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b A is to be transposed.
 * @param[in] diag				Specify whether matrix \b A is unit triangular.
 * @param[in] N					Number of rows/columns in matrix \b A.
 * @param[in] A					Buffer object storing matrix \b A.
 * @param[in] offa				Offset in number of elements for first element in matrix \b A.
 * @param[in] lda				Leading dimension of matrix \b A. It cannot be less
 *								than \b N
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff		Temporary cl_mem scratch buffer object which can hold a
 *								minimum of (1 + (N-1)*abs(incx)) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasStrmv() function otherwise.
 *
 * @ingroup TRMV
 */
clAmdBlasStatus
clAmdBlasDtrmv(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose trans,
    clAmdBlasDiag diag,
    size_t N,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
	cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Matrix-vector product with a triangular matrix and
 * float complex elements.
 *
 * Matrix-vector products:
 *   - \f$ X \leftarrow  A X \f$
 *   - \f$ X \leftarrow  A^T X \f$
 *
 * @param[in] order				Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b A is to be transposed.
 * @param[in] diag				Specify whether matrix \b A is unit triangular.
 * @param[in] N					Number of rows/columns in matrix \b A.
 * @param[in] A					Buffer object storing matrix \b A.
 * @param[in] offa				Offset in number of elements for first element in matrix \b A.
 * @param[in] lda				Leading dimension of matrix \b A. It cannot be less
 *								than \b N
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff		Temporary cl_mem scratch buffer object which can hold a
 *								minimum of (1 + (N-1)*abs(incx)) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return The same result as the clAmdBlasStrmv() function.
 * @ingroup TRMV
 */
clAmdBlasStatus 
clAmdBlasCtrmv(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose trans,
    clAmdBlasDiag diag,
    size_t N,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
	cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Matrix-vector product with a triangular matrix and
 * double complex elements.
 *
 * Matrix-vector products:
 *   - \f$ X \leftarrow  A X \f$
 *   - \f$ X \leftarrow  A^T X \f$
 *
 * @param[in] order				Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b A is to be transposed.
 * @param[in] diag				Specify whether matrix \b A is unit triangular.
 * @param[in] N					Number of rows/columns in matrix \b A.
 * @param[in] A					Buffer object storing matrix \b A.
 * @param[in] offa				Offset in number of elements for first element in matrix \b A.
 * @param[in] lda				Leading dimension of matrix \b A. It cannot be less
 *								than \b N
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff		Temporary cl_mem scratch buffer object which can hold a
 *								minimum of (1 + (N-1)*abs(incx)) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return The same result as the clAmdBlasDtrmv() function.
 * @ingroup TRMV
 */
clAmdBlasStatus 
clAmdBlasZtrmv(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose trans,
    clAmdBlasDiag diag,
    size_t N,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
	cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);


/*@}*/

/**
 * @defgroup TRSV TRSV  - TRiangular matrix Vector Solve
 * @ingroup BLAS2
 */
/*@{*/

/**
 * @brief solving triangular matrix problems with float elements.
 *
 * Matrix-vector products:
 *   - \f$ A X \leftarrow  X \f$
 *   - \f$ A^T X \leftarrow  X \f$
 *
 * @param[in] order				Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b A is to be transposed.
 * @param[in] diag				Specify whether matrix \b A is unit triangular.
 * @param[in] N					Number of rows/columns in matrix \b A.
 * @param[in] A					Buffer object storing matrix \b A.
 * @param[in] offa				Offset in number of elements for first element in matrix \b A.
 * @param[in] lda				Leading dimension of matrix \b A. It cannot be less
 *								than \b N
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - either \b N or \b incx is zero, or
 *     - the leading dimension is invalid;
 *   - \b clAmdBlasInvalidMemObject if either \b A or \b X object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup TRSV
 */
clAmdBlasStatus
clAmdBlasStrsv(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose trans,
    clAmdBlasDiag diag,
    size_t N,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @example example_strsv.c
 * Example of how to use the @ref clAmdBlasStrsv function.
 */


/**
 * @brief solving triangular matrix problems with double elements.
 *
 * Matrix-vector products:
 *   - \f$ A X \leftarrow  X \f$
 *   - \f$ A^T X \leftarrow  X \f$
 *
 * @param[in] order				Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b A is to be transposed.
 * @param[in] diag				Specify whether matrix \b A is unit triangular.
 * @param[in] N					Number of rows/columns in matrix \b A.
 * @param[in] A					Buffer object storing matrix \b A.
 * @param[in] offa				Offset in number of elements for first element in matrix \b A.
 * @param[in] lda				Leading dimension of matrix \b A. It cannot be less
 *								than \b N
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasStrsv() function otherwise.
 *
 * @ingroup TRSV
 */
clAmdBlasStatus
clAmdBlasDtrsv(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose trans,
    clAmdBlasDiag diag,
    size_t N,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);


/**
 * @brief solving triangular matrix problems with float-complex elements.
 *
 * Matrix-vector products:
 *   - \f$ A X \leftarrow  X \f$
 *   - \f$ A^T X \leftarrow  X \f$
 *
 * @param[in] order				Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b A is to be transposed.
 * @param[in] diag				Specify whether matrix \b A is unit triangular.
 * @param[in] N					Number of rows/columns in matrix \b A.
 * @param[in] A					Buffer object storing matrix \b A.
 * @param[in] offa				Offset in number of elements for first element in matrix \b A.
 * @param[in] lda				Leading dimension of matrix \b A. It cannot be less
 *								than \b N
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return The same result as the clAmdBlasStrsv() function.
 *
 * @ingroup TRSV
 */
clAmdBlasStatus
clAmdBlasCtrsv(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose trans,
    clAmdBlasDiag diag,
    size_t N,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);


/**
 * @brief solving triangular matrix problems with double-complex elements.
 *
 * Matrix-vector products:
 *   - \f$ A X \leftarrow  X \f$
 *   - \f$ A^T X \leftarrow  X \f$
 *
 * @param[in] order				Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b A is to be transposed.
 * @param[in] diag				Specify whether matrix \b A is unit triangular.
 * @param[in] N					Number of rows/columns in matrix \b A.
 * @param[in] A					Buffer object storing matrix \b A.
 * @param[in] offa				Offset in number of elements for first element in matrix \b A.
 * @param[in] lda				Leading dimension of matrix \b A. It cannot be less
 *								than \b N
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return The same result as the clAmdBlasDtrsv() function.
 *
 * @ingroup TRSV
 */
clAmdBlasStatus
clAmdBlasZtrsv(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose trans,
    clAmdBlasDiag diag,
    size_t N,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/*@}*/

/**
 * @defgroup GER GER   - GEneral matrix Rank 1 operation
 * @ingroup BLAS2
 */
/*@{*/

/**
 * @brief vector-vector product with float elements and
 * performs the rank 1 operation A
 *
 * Vector-vector products:
 *   - \f$ A \leftarrow \alpha X Y^T + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     specifies the scalar alpha.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset in number of elements for the first element in vector \b X.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] Y         Buffer object storing vector \b Y.
 * @param[in] offy      Offset in number of elements for the first element in vector \b Y.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[out] A 		Buffer object storing matrix \b A. On exit, A is
 *				        overwritten by the updated matrix.
 * @param[in] offa      Offset in number of elements for the first element in matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M when the
 *                      parameter is set to \b clAmdBlasColumnMajor.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - \b M, \b N or
 *	   - either \b incx or \b incy is zero, or
 *     - a leading dimension is invalid;
 *   - \b clAmdBlasInvalidMemObject if A, X, or Y object is invalid,
 *     or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfResources if you use image-based function implementation
 *     and no suitable scratch image available;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs to
 *     was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup GER
 */
clAmdBlasStatus
clAmdBlasSger(
    clAmdBlasOrder order,
    size_t M,
    size_t N,
    cl_float alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem A,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @example example_sger.c
 * Example of how to use the @ref clAmdBlasSger function.
 */


/**
 * @brief vector-vector product with double elements and
 * performs the rank 1 operation A
 *
 * Vector-vector products:
 *   - \f$ A \leftarrow \alpha X Y^T + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     specifies the scalar alpha.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset in number of elements for the first element in vector \b X.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] Y         Buffer object storing vector \b Y.
 * @param[in] offy      Offset in number of elements for the first element in vector \b Y.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[out] A 		Buffer object storing matrix \b A. On exit, A is
 *				        overwritten by the updated matrix.
 * @param[in] offa      Offset in number of elements for the first element in matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M when the
 *                      parameter is set to \b clAmdBlasColumnMajor.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasSger() function otherwise.
 *
 * @ingroup GER
 */
clAmdBlasStatus
clAmdBlasDger(
    clAmdBlasOrder order,
    size_t M,
    size_t N,
    cl_double alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem A,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/*@}*/

/**
 * @defgroup GERU GERU  - GEneral matrix Rank 1 operation
 * @ingroup BLAS2
 */
/*@{*/

/**
 * @brief vector-vector product with float complex elements and
 * performs the rank 1 operation A
 *
 * Vector-vector products:
 *   - \f$ A \leftarrow \alpha X Y^T + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     specifies the scalar alpha.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset in number of elements for the first element in vector \b X.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] Y         Buffer object storing vector \b Y.
 * @param[in] offy      Offset in number of elements for the first element in vector \b Y.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[out] A 		Buffer object storing matrix \b A. On exit, A is
 *				        overwritten by the updated matrix.
 * @param[in] offa      Offset in number of elements for the first element in matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M when the
 *                      parameter is set to \b clAmdBlasColumnMajor.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - \b M, \b N or
 *	   - either \b incx or \b incy is zero, or
 *     - a leading dimension is invalid;
 *   - \b clAmdBlasInvalidMemObject if A, X, or Y object is invalid,
 *     or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfResources if you use image-based function implementation
 *     and no suitable scratch image available;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs to
 *     was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup GERU
 */
clAmdBlasStatus
clAmdBlasCgeru(
    clAmdBlasOrder order,
    size_t M,
    size_t N,
    cl_float2 alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem A ,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief vector-vector product with double complex elements and
 * performs the rank 1 operation A
 *
 * Vector-vector products:
 *   - \f$ A \leftarrow \alpha X Y^T + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     specifies the scalar alpha.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset in number of elements for the first element in vector \b X.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] Y         Buffer object storing vector \b Y.
 * @param[in] offy      Offset in number of elements for the first element in vector \b Y.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[out] A		   Buffer object storing matrix \b A. On exit, A is
 *				        overwritten by the updated matrix.
 * @param[in] offa      Offset in number of elements for the first element in matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M when the
 *                      parameter is set to \b clAmdBlasColumnMajor.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasCgeru() function otherwise.
 *
 * @ingroup GERU
 */
clAmdBlasStatus
clAmdBlasZgeru(
    clAmdBlasOrder order,
    size_t M,
    size_t N,
    cl_double2 alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem A,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/*@}*/

/**
 * @defgroup GERC GERC  - GEneral matrix Rank 1 operation
 * @ingroup BLAS2
 */
/*@{*/

/**
 * @brief vector-vector product with float complex elements and
 * performs the rank 1 operation A
 *
 * Vector-vector products:
 *   - \f$ A \leftarrow \alpha X Y^H + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     specifies the scalar alpha.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset in number of elements for the first element in vector \b X.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] Y         Buffer object storing vector \b Y.
 * @param[in] offy      Offset in number of elements for the first element in vector \b Y.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[out] A 	    Buffer object storing matrix \b A. On exit, A is
 *				        overwritten by the updated matrix.
 * @param[in] offa      Offset in number of elements for the first element in matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M when the
 *                      parameter is set to \b clAmdBlasColumnMajor.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - \b M, \b N or
 *	   - either \b incx or \b incy is zero, or
 *     - a leading dimension is invalid;
 *   - \b clAmdBlasInvalidMemObject if A, X, or Y object is invalid,
 *     or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfResources if you use image-based function implementation
 *     and no suitable scratch image available;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs to
 *     was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup GERC
 */

clAmdBlasStatus
clAmdBlasCgerc(
    clAmdBlasOrder order,
    size_t M,
    size_t N,
    cl_float2 alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem A ,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief vector-vector product with double complex elements and
 * performs the rank 1 operation A
 *
 * Vector-vector products:
 *   - \f$ A \leftarrow \alpha X Y^H + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     specifies the scalar alpha.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset in number of elements for the first element in vector \b X.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] Y         Buffer object storing vector \b Y.
 * @param[in] offy      Offset in number of elements for the first element in vector \b Y.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[out] A		Buffer object storing matrix \b A. On exit, A is
 *				        overwritten by the updated matrix.
 * @param[in] offa      Offset in number of elements for the first element in matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M when the
 *                      parameter is set to \b clAmdBlasColumnMajor.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasCgerc() function otherwise.
 *
 * @ingroup GERC
 */
clAmdBlasStatus
clAmdBlasZgerc(
    clAmdBlasOrder order,
    size_t M,
    size_t N,
    cl_double2 alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem A,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);


/*@}*/

/**
 * @defgroup SYR SYR   - SYmmetric Rank 1 update
 *
 * The Level 2 Basic Linear Algebra Subprograms are functions that perform
 * symmetric rank 1 update operations.
  * @ingroup BLAS2
 */

/*@{*/
/**
 * @brief Symmetric rank 1 operation with a general triangular matrix and
 * float elements.
 *
 * Symmetric rank 1 operation:
 *   - \f$ A \leftarrow \alpha x x^T + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] A 	    Buffer object storing matrix \b A.
 * @param[in] offa      Offset of first element of matrix \b A in buffer object.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - either \b incx is zero, or
 *     - the leading dimension is invalid;
 *   - \b clAmdBlasInvalidMemObject if either \b A, \b X object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup SYR
 */
clAmdBlasStatus
clAmdBlasSsyr(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    size_t N,
    cl_float alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem A,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events);

/**
 * @brief Symmetric rank 1 operation with a general triangular matrix and
 * double elements.
 *
 * Symmetric rank 1 operation:
 *   - \f$ A \leftarrow \alpha x x^T + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] A		Buffer object storing matrix \b A.
 * @param[in] offa      Offset of first element of matrix \b A in buffer object.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasSsyr() function otherwise.
 *
 * @ingroup SYR
 */

clAmdBlasStatus
clAmdBlasDsyr(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    size_t N,
    cl_double alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem A,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events);
/*@}*/


/**
 * @defgroup HER HER   - HErmitian Rank 1 operation 
 *
 * The Level 2 Basic Linear Algebra Subprogram functions that perform
 * hermitian rank 1 operations.
 * @ingroup BLAS2
 */

/*@{*/
/**
 * @brief hermitian rank 1 operation with a general triangular matrix and
 * float-complex elements.
 *
 * hermitian rank 1 operation:
 *   - \f$ A \leftarrow \alpha X X^H + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A (a scalar float value)
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset in number of elements for the first element in vector \b X.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] A		Buffer object storing matrix \b A.
 * @param[in] offa      Offset in number of elements for the first element in matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - either \b incx is zero, or
 *     - the leading dimension is invalid;
 *   - \b clAmdBlasInvalidMemObject if either \b A, \b X object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup HER
 */
clAmdBlasStatus
clAmdBlasCher(
	clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    size_t N,
    cl_float alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem A,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events);
/**
 * @example example_cher.c
 * Example of how to use the @ref clAmdBlasCher function.
 */

/**
 * @brief hermitian rank 1 operation with a general triangular matrix and
 * double-complex elements.
 *
 * hermitian rank 1 operation:
 *   - \f$ A \leftarrow \alpha X X^H + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A (a scalar double value)
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset in number of elements for the first element in vector \b X.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] A		Buffer object storing matrix \b A.
 * @param[in] offa      Offset in number of elements for the first element in matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasCher() function otherwise.
 *
 * @ingroup HER
 */
clAmdBlasStatus
clAmdBlasZher(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    size_t N,
    cl_double alpha,
    const cl_mem X,
    size_t offx,
    int incx,
	cl_mem A,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events);
/*@}*/

/**
 * @defgroup SYR2 SYR2  - SYmmetric Rank 2 update
 *
 * The Level 2 Basic Linear Algebra Subprograms are functions that perform
 * symmetric rank 2 update operations.
  * @ingroup BLAS2
 */

/*@{*/
/**
 * @brief Symmetric rank 2 operation with a general triangular matrix and
 * float elements.
 *
 * Symmetric rank 2 operation:
 *   - \f$ A \leftarrow \alpha x y^T + \alpha y x^T + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] Y         Buffer object storing vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[out] A 	    Buffer object storing matrix \b A.
 * @param[in] offa      Offset of first element of matrix \b A in buffer object.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - either \b N is zero, or
 *     - either \b incx or \b incy is zero, or
 *     - the leading dimension is invalid;
 *   - \b clAmdBlasInvalidMemObject if either \b A, \b X, or \b Y object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup SYR2
 */

clAmdBlasStatus
clAmdBlasSsyr2(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    size_t N,
    cl_float alpha,
    const cl_mem X,
    size_t offx,
    int  incx,
	const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem A,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events);

/**
 * @brief Symmetric rank 2 operation with a general triangular matrix and
 * double elements.
 *
 * Symmetric rank 2 operation:
 *   - \f$ A \leftarrow \alpha x y^T + \alpha y x^T + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] Y         Buffer object storing vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[out] A 	    Buffer object storing matrix \b A.
 * @param[in] offa      Offset of first element of matrix \b A in buffer object.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - either \b N is zero, or
 *     - either \b incx or \b incy is zero, or
 *     - the leading dimension is invalid;
 *   - \b clAmdBlasInvalidMemObject if either \b A, \b X, or \b Y object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup SYR2
 */

clAmdBlasStatus
clAmdBlasDsyr2(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    size_t N,
    cl_double alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem A,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events);

/*@}*/

/**
 * @defgroup HER2 HER2  - HErmitian Rank 2 update
 *
 * The Level 2 Basic Linear Algebra Subprograms are functions that perform
 * hermitian rank 2 update operations.
 * @ingroup BLAS2
 */

/*@{*/
/**
 * @brief Hermitian rank 2 operation with a general triangular matrix and
 * float-compelx elements.
 *
 * Hermitian rank 2 operation:
 *   - \f$ A \leftarrow \alpha X Y^H + \overline{ \alpha } Y X^H + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset in number of elements for the first element in vector \b X.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] Y         Buffer object storing vector \b Y.
 * @param[in] offy      Offset in number of elements for the first element in vector \b Y.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[out] A		Buffer object storing matrix \b A.
 * @param[in] offa      Offset in number of elements for the first element in matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - either \b N is zero, or
 *     - either \b incx or \b incy is zero, or
 *     - the leading dimension is invalid;
 *   - \b clAmdBlasInvalidMemObject if either \b A, \b X, or \b Y object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup HER2
 */
clAmdBlasStatus
clAmdBlasCher2(
	clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    size_t N,
    cl_float2 alpha,
    const cl_mem X,
    size_t offx,
    int incx,
	const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem A,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events);


/**
* @brief Hermitian rank 2 operation with a general triangular matrix and
 * double-compelx elements.
 *
 * Hermitian rank 2 operation:
 *   - \f$ A \leftarrow \alpha X Y^H + \overline{ \alpha } Y X^H + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset in number of elements for the first element in vector \b X.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] Y         Buffer object storing vector \b Y.
 * @param[in] offy      Offset in number of elements for the first element in vector \b Y.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[out] A		Buffer object storing matrix \b A.
 * @param[in] offa      Offset in number of elements for the first element in matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasCher2() function otherwise.
 *
 * @ingroup HER2
 */
clAmdBlasStatus
clAmdBlasZher2(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    size_t N,
    cl_double2 alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
	cl_mem A,
    size_t offa,
    size_t lda,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events);

/**
 * @example example_zher2.c
 * Example of how to use the @ref clAmdBlasZher2 function.
 */

/*@}*/

/**
 * @defgroup TPMV TPMV  - Triangular Packed Matrix-Vector multiply
 * @ingroup BLAS2
 */
/*@{*/

/**
 * @brief Matrix-vector product with a packed triangular matrix and
 * float elements.
 *
 * Matrix-vector products:
 *   - \f$ X \leftarrow  A X \f$
 *   - \f$ X \leftarrow  A^T X \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b AP is to be transposed.
 * @param[in] diag				Specify whether matrix \b AP is unit triangular.
 * @param[in] N					Number of rows/columns in matrix \b A.
 * @param[in] AP				Buffer object storing matrix \b AP in packed format.
 * @param[in] offa				Offset in number of elements for first element in matrix \b AP.
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff		Temporary cl_mem scratch buffer object which can hold a
 *								minimum of (1 + (N-1)*abs(incx)) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - either \b N or \b incx is zero
 *   - \b clAmdBlasInvalidMemObject if either \b AP or \b X object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup TPMV
 */
clAmdBlasStatus 
clAmdBlasStpmv(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose trans,
    clAmdBlasDiag diag,
    size_t N,
    const cl_mem AP,
    size_t offa,
    cl_mem X,
    size_t offx,
    int incx,
	cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @example example_stpmv.c
 * Example of how to use the @ref clAmdBlasStpmv function.
 */

/**
 * @brief Matrix-vector product with a packed triangular matrix and
 * double elements.
 *
 * Matrix-vector products:
 *   - \f$ X \leftarrow  A X \f$
 *   - \f$ X \leftarrow  A^T X \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b AP is to be transposed.
 * @param[in] diag				Specify whether matrix \b AP is unit triangular.
 * @param[in] N					Number of rows/columns in matrix \b AP.
 * @param[in] AP				Buffer object storing matrix \b AP in packed format.
 * @param[in] offa				Offset in number of elements for first element in matrix \b AP.
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff		Temporary cl_mem scratch buffer object which can hold a
 *								minimum of (1 + (N-1)*abs(incx)) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasStpmv() function otherwise.
 *
 * @ingroup TPMV
 */
clAmdBlasStatus
clAmdBlasDtpmv(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose trans,
    clAmdBlasDiag diag,
    size_t N,
    const cl_mem AP,
    size_t offa,
    cl_mem X,
    size_t offx,
    int incx,
	cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
  * @brief Matrix-vector product with a packed triangular matrix and
 * float-complex elements.
 *
 * Matrix-vector products:
 *   - \f$ X \leftarrow  A X \f$
 *   - \f$ X \leftarrow  A^T X \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b AP is to be transposed.
 * @param[in] diag				Specify whether matrix \b AP is unit triangular.
 * @param[in] N					Number of rows/columns in matrix \b AP.
 * @param[in] AP				Buffer object storing matrix \b AP in packed format.
 * @param[in] offa				Offset in number of elements for first element in matrix \b AP.
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff		Temporary cl_mem scratch buffer object which can hold a
 *								minimum of (1 + (N-1)*abs(incx)) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return The same result as the clAmdBlasStpmv() function.
 * @ingroup TPMV
 */
clAmdBlasStatus 
clAmdBlasCtpmv(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose trans,
    clAmdBlasDiag diag,
    size_t N,
    const cl_mem AP,
    size_t offa,
    cl_mem X,
    size_t offx,
    int incx,
	cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Matrix-vector product with a packed triangular matrix and
 * double-complex elements.
 *
 * Matrix-vector products:
 *   - \f$ X \leftarrow  A X \f$
 *   - \f$ X \leftarrow  A^T X \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b AP is to be transposed.
 * @param[in] diag				Specify whether matrix \b AP is unit triangular.
 * @param[in] N					Number of rows/columns in matrix \b AP.
 * @param[in] AP				Buffer object storing matrix \b AP in packed format.
 * @param[in] offa				Offset in number of elements for first element in matrix \b AP.
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff		Temporary cl_mem scratch buffer object which can hold a
 *								minimum of (1 + (N-1)*abs(incx)) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return The same result as the clAmdBlasDtpmv() function.
 * @ingroup TPMV
 */
clAmdBlasStatus 
clAmdBlasZtpmv(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose trans,
    clAmdBlasDiag diag,
    size_t N,
    const cl_mem AP,
    size_t offa,
    cl_mem X,
    size_t offx,
    int incx,
	cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/*@}*/



/**
 * @defgroup TPSV TPSV  - Triangular Packed matrix Vector Solve 
 * @ingroup BLAS2
 */
/*@{*/

/**
 * @brief solving triangular packed matrix problems with float elements.
 *
 * Matrix-vector products:
 *   - \f$ A X \leftarrow  X \f$
 *   - \f$ A^T X \leftarrow  X \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo              The triangle in matrix being referenced.
 * @param[in] trans             How matrix \b A is to be transposed.
 * @param[in] diag              Specify whether matrix \b A is unit triangular.
 * @param[in] N                 Number of rows/columns in matrix \b A.
 * @param[in] A                 Buffer object storing matrix in packed format.\b A.
 * @param[in] offa              Offset in number of elements for first element in matrix \b A.
 * @param[out] X                Buffer object storing vector \b X.
 * @param[in] offx              Offset in number of elements for first element in vector \b X.
 * @param[in] incx              Increment for the elements of \b X. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - either \b N or \b incx is zero, or
 *     - the leading dimension is invalid;
 *   - \b clAmdBlasInvalidMemObject if either \b A or \b X object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup TPSV
 */

clAmdBlasStatus
clAmdBlasStpsv(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose trans,
    clAmdBlasDiag diag,
    size_t N,
    const cl_mem A,
    size_t offa,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @example example_stpsv.c
 * Example of how to use the @ref clAmdBlasStpsv function.
 */

/**
 * @brief solving triangular packed matrix problems with double elements.
 *
 * Matrix-vector products:
 *   - \f$ A X \leftarrow  X \f$
 *   - \f$ A^T X \leftarrow  X \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo              The triangle in matrix being referenced.
 * @param[in] trans             How matrix \b A is to be transposed.
 * @param[in] diag              Specify whether matrix \b A is unit triangular.
 * @param[in] N                 Number of rows/columns in matrix \b A.
 * @param[in] A                 Buffer object storing matrix in packed format.\b A.
 * @param[in] offa              Offset in number of elements for first element in matrix \b A.
 * @param[out] X                Buffer object storing vector \b X.
 * @param[in] offx              Offset in number of elements for first element in vector \b X.
 * @param[in] incx              Increment for the elements of \b X. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - either \b N or \b incx is zero, or
 *     - the leading dimension is invalid;
 *   - \b clAmdBlasInvalidMemObject if either \b A or \b X object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup TPSV
 */

clAmdBlasStatus
clAmdBlasDtpsv(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose trans,
    clAmdBlasDiag diag,
    size_t N,
    const cl_mem A,
    size_t offa,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief solving triangular packed matrix problems with float complex elements.
 *
 * Matrix-vector products:
 *   - \f$ A X \leftarrow  X \f$
 *   - \f$ A^T X \leftarrow  X \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo              The triangle in matrix being referenced.
 * @param[in] trans             How matrix \b A is to be transposed.
 * @param[in] diag              Specify whether matrix \b A is unit triangular.
 * @param[in] N                 Number of rows/columns in matrix \b A.
 * @param[in] A                 Buffer object storing matrix in packed format.\b A.
 * @param[in] offa              Offset in number of elements for first element in matrix \b A.
 * @param[out] X                Buffer object storing vector \b X.
 * @param[in] offx              Offset in number of elements for first element in vector \b X.
 * @param[in] incx              Increment for the elements of \b X. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - either \b N or \b incx is zero, or
 *     - the leading dimension is invalid;
 *   - \b clAmdBlasInvalidMemObject if either \b A or \b X object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup TPSV
 */

clAmdBlasStatus
clAmdBlasCtpsv(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose trans,
    clAmdBlasDiag diag,
    size_t N,
    const cl_mem A,
    size_t offa,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief solving triangular packed matrix problems with double complex elements.
 *
 * Matrix-vector products:
 *   - \f$ A X \leftarrow  X \f$
 *   - \f$ A^T X \leftarrow  X \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo              The triangle in matrix being referenced.
 * @param[in] trans             How matrix \b A is to be transposed.
 * @param[in] diag              Specify whether matrix \b A is unit triangular.
 * @param[in] N                 Number of rows/columns in matrix \b A.
 * @param[in] A                 Buffer object storing matrix in packed format.\b A.
 * @param[in] offa              Offset in number of elements for first element in matrix \b A.
 * @param[out] X                Buffer object storing vector \b X.
 * @param[in] offx              Offset in number of elements for first element in vector \b X.
 * @param[in] incx              Increment for the elements of \b X. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - either \b N or \b incx is zero, or
 *     - the leading dimension is invalid;
 *   - \b clAmdBlasInvalidMemObject if either \b A or \b X object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup TPSV
 */

clAmdBlasStatus
clAmdBlasZtpsv(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose trans,
    clAmdBlasDiag diag,
    size_t N,
    const cl_mem A,
    size_t offa,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/*@}*/


/**
 * @defgroup SPMV SPMV  - Symmetric Packed Matrix Vector multiply
 * @ingroup BLAS2
 */

/*@{*/

/**
 * @brief Matrix-vector product with a symmetric packed-matrix and float elements.
 *
 * Matrix-vector products:
 * - \f$ Y \leftarrow \alpha A X + \beta Y \f$
 *
 * @param[in] order     Row/columns order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of rows and columns in matrix \b AP.
 * @param[in] alpha     The factor of matrix \b AP.
 * @param[in] AP        Buffer object storing matrix \b AP.
 * @param[in] offa		Offset in number of elements for first element in matrix \b AP.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of vector \b X. It cannot be zero.
 * @param[in] beta      The factor of vector \b Y.
 * @param[out] Y        Buffer object storing vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of vector \b Y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - either \b incx or \b incy is zero, or
 *     - the matrix sizes or the vector sizes along with the increments lead to
 *       accessing outsize of any of the buffers;
 *   - \b clAmdBlasInvalidMemObject if either \b AP, \b X, or \b Y object is
 *     invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs to
 *     was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup SPMV
 */
clAmdBlasStatus
clAmdBlasSspmv(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    size_t N,
    cl_float alpha,
    const cl_mem AP,
    size_t offa,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_float beta,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @example example_sspmv.c
 * This is an example of how to use the @ref clAmdBlasSspmv function.
 */

/**
 * @brief Matrix-vector product with a symmetric packed-matrix and double elements.
 *
 * Matrix-vector products:
 * - \f$ Y \leftarrow \alpha A X + \beta Y \f$
 *
 * @param[in] order     Row/columns order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of rows and columns in matrix \b AP.
 * @param[in] alpha     The factor of matrix \b AP.
 * @param[in] AP        Buffer object storing matrix \b AP.
 * @param[in] offa		Offset in number of elements for first element in matrix \b AP.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of vector \b X. It cannot be zero.
 * @param[in] beta      The factor of vector \b Y.
 * @param[out] Y        Buffer object storing vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of vector \b Y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasSspmv() function otherwise.
 *
 * @ingroup SPMV
 */
clAmdBlasStatus
clAmdBlasDspmv(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    size_t N,
    cl_double alpha,
    const cl_mem AP,
    size_t offa,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_double beta,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/*@}*/



/**
 * @defgroup HPMV HPMV  - Hermitian Packed Matrix-Vector multiplication
 * @ingroup BLAS2
 */

/*@{*/

/**
 * @brief Matrix-vector product with a packed hermitian matrix and float-complex elements.
 *
 * Matrix-vector products:
 * - \f$ Y \leftarrow \alpha A X + \beta Y \f$
 *
 * @param[in] order     Row/columns order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of rows and columns in matrix \b AP.
 * @param[in] alpha     The factor of matrix \b AP.
 * @param[in] AP        Buffer object storing packed matrix \b AP.
 * @param[in] offa		Offset in number of elements for first element in matrix \b AP.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of vector \b X. It cannot be zero.
 * @param[in] beta      The factor of vector \b Y.
 * @param[out] Y        Buffer object storing vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of vector \b Y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - either \b incx or \b incy is zero, or
 *     - the matrix sizes or the vector sizes along with the increments lead to
 *       accessing outsize of any of the buffers;
 *   - \b clAmdBlasInvalidMemObject if either \b AP, \b X, or \b Y object is
 *     invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs to
 *     was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup HPMV
 */
clAmdBlasStatus
clAmdBlasChpmv(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    size_t N,
    cl_float2 alpha,
    const cl_mem AP,
    size_t offa,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_float2 beta,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @example example_chpmv.c
 * This is an example of how to use the @ref clAmdBlasChpmv function.
 */


/**
 * @brief Matrix-vector product with a packed hermitian matrix and double-complex elements.
 *
 * Matrix-vector products:
 * - \f$ Y \leftarrow \alpha A X + \beta Y \f$
 *
 * @param[in] order     Row/columns order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of rows and columns in matrix \b AP.
 * @param[in] alpha     The factor of matrix \b AP.
 * @param[in] AP        Buffer object storing packed matrix \b AP.
 * @param[in] offa		Offset in number of elements for first element in matrix \b AP.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of vector \b X. It cannot be zero.
 * @param[in] beta      The factor of vector \b Y.
 * @param[out] Y        Buffer object storing vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of vector \b Y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasChpmv() function otherwise.
 *
 * @ingroup HPMV
 */
clAmdBlasStatus
clAmdBlasZhpmv(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    size_t N,
    cl_double2 alpha,
    const cl_mem AP,
    size_t offa,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_double2 beta,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/*@}*/


/**
 * @defgroup SPR SPR   - Symmetric Packed matrix Rank 1 update
 *
 * The Level 2 Basic Linear Algebra Subprograms are functions that perform
 * symmetric rank 1 update operations on packed matrix
 * @ingroup BLAS2
 */

/*@{*/
/**
 * @brief Symmetric rank 1 operation with a general triangular packed-matrix and
 * float elements.
 *
 * Symmetric rank 1 operation:
 *   - \f$ A \leftarrow \alpha X X^T + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] AP 	    Buffer object storing packed-matrix \b AP.
 * @param[in] offa      Offset of first element of matrix \b AP in buffer object.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - either \b incx is zero
 *   - \b clAmdBlasInvalidMemObject if either \b AP, \b X object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup SPR
 */
clAmdBlasStatus
clAmdBlasSspr(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    size_t N,
    cl_float alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem AP,
    size_t offa,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events);
/**
 * @example example_sspr.c
 * Example of how to use the @ref clAmdBlasSspr function.
 */

/**
 * @brief Symmetric rank 1 operation with a general triangular packed-matrix and
 * double elements.
 *
 * Symmetric rank 1 operation:
 *   - \f$ A \leftarrow \alpha X X^T + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] AP 	    Buffer object storing packed-matrix \b AP.
 * @param[in] offa      Offset of first element of matrix \b AP in buffer object.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasSspr() function otherwise.
 *
 * @ingroup SPR
 */

clAmdBlasStatus
clAmdBlasDspr(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    size_t N,
    cl_double alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem AP,
    size_t offa,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events);
/*@}*/

/**
 * @defgroup HPR HPR   - Hermitian Packed matrix Rank 1 update
 *
 * The Level 2 Basic Linear Algebra Subprogram functions that perform
 * hermitian rank 1 operations on packed matrix
 * @ingroup BLAS2
 */

/*@{*/
/**
 * @brief hermitian rank 1 operation with a general triangular packed-matrix and
 * float-complex elements.
 *
 * hermitian rank 1 operation:
 *   - \f$ A \leftarrow \alpha X X^H + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A (a scalar float value)
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset in number of elements for the first element in vector \b X.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] AP 	    Buffer object storing matrix \b AP.
 * @param[in] offa      Offset in number of elements for the first element in matrix \b AP.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - \b N is zero, or
 *     - either \b incx is zero
 *   - \b clAmdBlasInvalidMemObject if either \b AP, \b X object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup HPR
 */
clAmdBlasStatus
clAmdBlasChpr(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    size_t N,
    cl_float alpha,
    const cl_mem X,
    size_t offx,
    int  incx,
    cl_mem AP,
    size_t offa,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events);
/**
 * @example example_chpr.c
 * Example of how to use the @ref clAmdBlasChpr function.
 */

/**
 * @brief hermitian rank 1 operation with a general triangular packed-matrix and
 * double-complex elements.
 *
 * hermitian rank 1 operation:
 *   - \f$ A \leftarrow \alpha X X^H + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A (a scalar float value)
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset in number of elements for the first element in vector \b X.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[out] AP 	    Buffer object storing matrix \b AP.
 * @param[in] offa      Offset in number of elements for the first element in matrix \b AP.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasChpr() function otherwise.
 *
 * @ingroup HPR
 */
clAmdBlasStatus
clAmdBlasZhpr(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    size_t N,
    cl_double alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem AP,
    size_t offa,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events);
/*@}*/

/**
 * @defgroup SPR2 SPR2  - Symmetric Packed matrix Rank 2 update
 *
 * The Level 2 Basic Linear Algebra Subprograms are functions that perform
 * symmetric rank 2 update operations on packed matrices
 * @ingroup BLAS2
 */

/*@{*/
/**
 * @brief Symmetric rank 2 operation with a general triangular packed-matrix and
 * float elements.
 *
 * Symmetric rank 2 operation:
 *   - \f$ A \leftarrow \alpha X Y^T + \alpha Y X^T + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] Y         Buffer object storing vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[out] AP		Buffer object storing packed-matrix \b AP.
 * @param[in] offa      Offset of first element of matrix \b AP in buffer object.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - either \b N is zero, or
 *     - either \b incx or \b incy is zero
 *   - \b clAmdBlasInvalidMemObject if either \b AP, \b X, or \b Y object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup SPR2
 */

clAmdBlasStatus
clAmdBlasSspr2(
	clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    size_t N,
    cl_float alpha,
    const cl_mem X,
    size_t offx,
    int incx,
	const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem AP,
    size_t offa,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events);
/**
 * @example example_sspr2.c
 * Example of how to use the @ref clAmdBlasSspr2 function.
 */

/**
 * @brief Symmetric rank 2 operation with a general triangular packed-matrix and
 * double elements.
 *
 * Symmetric rank 2 operation:
 *   - \f$ A \leftarrow \alpha X Y^T + \alpha Y X^T + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] Y         Buffer object storing vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[out] AP		Buffer object storing packed-matrix \b AP.
 * @param[in] offa      Offset of first element of matrix \b AP in buffer object.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasSspr2() function otherwise.
 *
 * @ingroup SPR2
 */

clAmdBlasStatus
clAmdBlasDspr2(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    size_t N,
    cl_double alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
	cl_mem AP,
    size_t offa,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events);
/*@}*/

/**
 * @defgroup HPR2 HPR2  - Hermitian Packed matrix Rank 2 update
 *
 * The Level 2 Basic Linear Algebra Subprograms are functions that perform
 * hermitian rank 2 update operations on packed matrices
 * @ingroup BLAS2
 */

/*@{*/
/**
 * @brief Hermitian rank 2 operation with a general triangular packed-matrix and
 * float-compelx elements.
 *
 * Hermitian rank 2 operation:
 *   - \f$ A \leftarrow \alpha X Y^H + \conjg( alpha ) Y X^H + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset in number of elements for the first element in vector \b X.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] Y         Buffer object storing vector \b Y.
 * @param[in] offy      Offset in number of elements for the first element in vector \b Y.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[out] AP		Buffer object storing packed-matrix \b AP.
 * @param[in] offa      Offset in number of elements for the first element in matrix \b AP.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - either \b N is zero, or
 *     - either \b incx or \b incy is zero
 *   - \b clAmdBlasInvalidMemObject if either \b AP, \b X, or \b Y object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup HPR2
 */
clAmdBlasStatus
clAmdBlasChpr2(
	clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    size_t N,
    cl_float2 alpha,
    const cl_mem X,
    size_t offx,
    int incx,
	const cl_mem Y,
    size_t offy,
    int incy,
    cl_mem AP,
    size_t offa,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events);


/**
 * @brief Hermitian rank 2 operation with a general triangular packed-matrix and
 * double-compelx elements.
 *
 * Hermitian rank 2 operation:
 *   - \f$ A \leftarrow \alpha X Y^H + \conjg( alpha ) Y X^H + A \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of columns in matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset in number of elements for the first element in vector \b X.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] Y         Buffer object storing vector \b Y.
 * @param[in] offy      Offset in number of elements for the first element in vector \b Y.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[out] AP		Buffer object storing packed-matrix \b AP.
 * @param[in] offa      Offset in number of elements for the first element in matrix \b AP.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasChpr2() function otherwise.
 *
 * @ingroup HPR2
 */
clAmdBlasStatus
clAmdBlasZhpr2(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    size_t N,
    cl_double2 alpha,
    const cl_mem X,
    size_t offx,
    int incx,
    const cl_mem Y,
    size_t offy,
    int incy,
	cl_mem AP,
    size_t offa,
    cl_uint numCommandQueues,
    cl_command_queue* commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event* eventWaitList,
    cl_event* events);

/**
 * @example example_zhpr2.c
 * Example of how to use the @ref clAmdBlasZhpr2 function.
 */
/*@}*/



/**
 * @defgroup GBMV GBMV  - General Banded Matrix-Vector multiplication
 * @ingroup BLAS2
 */
/*@{*/

/**
 * @brief Matrix-vector product with a general rectangular banded matrix and
 * float elements.
 *
 * Matrix-vector products:
 *   - \f$ Y \leftarrow \alpha A X + \beta Y \f$
 *   - \f$ Y \leftarrow \alpha A^T X + \beta Y \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] trans     How matrix \b A is to be transposed.
 * @param[in] M         Number of rows in banded matrix \b A.
 * @param[in] N         Number of columns in banded matrix \b A.
 * @param[in] KL        Number of sub-diagonals in banded matrix \b A.
 * @param[in] KU        Number of super-diagonals in banded matrix \b A.
 * @param[in] alpha     The factor of banded matrix \b A.
 * @param[in] A         Buffer object storing banded matrix \b A.
 * @param[in] offa      Offset in number of elements for the first element in banded matrix \b A.
 * @param[in] lda       Leading dimension of banded matrix \b A. It cannot be less
 *                      than ( \b KL + \b KU + 1 )
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] beta      The factor of the vector \b Y.
 * @param[out] Y        Buffer object storing the vector \b y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - either \b M or \b N is zero, or
 *     - KL is greater than \b M - 1, or
 *     - KU is greater than \b N - 1, or
 *     - either \b incx or \b incy is zero, or
 *     - any of the leading dimensions is invalid;
 *     - the matrix size or the vector sizes along with the increments lead to
 *       accessing outside of any of the buffers;
 *   - \b clAmdBlasInvalidMemObject if either \b A, \b X, or \b Y object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup GBMV
 */
clAmdBlasStatus
clAmdBlasSgbmv( 
    clAmdBlasOrder order, 
    clAmdBlasTranspose trans, 
    size_t M, 
    size_t N, 
    size_t KL, 
    size_t KU, 
    cl_float alpha, 
    const cl_mem A, 
    size_t offa, 
    size_t lda, 
    const cl_mem X, 
    size_t offx, 
    int incx, 
    cl_float beta, 
    cl_mem Y, 
    size_t offy, 
    int incy, 
    cl_uint numCommandQueues, 
    cl_command_queue *commandQueues, 
    cl_uint numEventsInWaitList, 
    const cl_event *eventWaitList, 
    cl_event *events);
/**
 * @example example_sgbmv.c
 * Example of how to use the @ref clAmdBlasSgbmv function.
 */


/**
 * @brief Matrix-vector product with a general rectangular banded matrix and
 * double elements.
 *
 * Matrix-vector products:
 *   - \f$ Y \leftarrow \alpha A X + \beta Y \f$
 *   - \f$ Y \leftarrow \alpha A^T X + \beta Y \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] trans     How matrix \b A is to be transposed.
 * @param[in] M         Number of rows in banded matrix \b A.
 * @param[in] N         Number of columns in banded matrix \b A.
 * @param[in] KL        Number of sub-diagonals in banded matrix \b A.
 * @param[in] KU        Number of super-diagonals in banded matrix \b A.
 * @param[in] alpha     The factor of banded matrix \b A.
 * @param[in] A         Buffer object storing banded matrix \b A.
 * @param[in] offa      Offset in number of elements for the first element in banded matrix \b A.
 * @param[in] lda       Leading dimension of banded matrix \b A. It cannot be less
 *                      than ( \b KL + \b KU + 1 )
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] beta      The factor of the vector \b Y.
 * @param[out] Y        Buffer object storing the vector \b y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasSgbmv() function otherwise.
 *
 * @ingroup GBMV
 */
clAmdBlasStatus
clAmdBlasDgbmv(
    clAmdBlasOrder order,
    clAmdBlasTranspose trans,
    size_t M,
    size_t N,
    size_t KL,
    size_t KU,
    cl_double alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_double beta,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);


/**
 * @brief Matrix-vector product with a general rectangular banded matrix and
 * float-complex elements.
 *
 * Matrix-vector products:
 *   - \f$ Y \leftarrow \alpha A X + \beta Y \f$
 *   - \f$ Y \leftarrow \alpha A^T X + \beta Y \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] trans     How matrix \b A is to be transposed.
 * @param[in] M         Number of rows in banded matrix \b A.
 * @param[in] N         Number of columns in banded matrix \b A.
 * @param[in] KL        Number of sub-diagonals in banded matrix \b A.
 * @param[in] KU        Number of super-diagonals in banded matrix \b A.
 * @param[in] alpha     The factor of banded matrix \b A.
 * @param[in] A         Buffer object storing banded matrix \b A.
 * @param[in] offa      Offset in number of elements for the first element in banded matrix \b A.
 * @param[in] lda       Leading dimension of banded matrix \b A. It cannot be less
 *                      than ( \b KL + \b KU + 1 )
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] beta      The factor of the vector \b Y.
 * @param[out] Y        Buffer object storing the vector \b y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return The same result as the clAmdBlasSgbmv() function.
 *
 * @ingroup GBMV
 */    
clAmdBlasStatus
clAmdBlasCgbmv( 
    clAmdBlasOrder order, 
    clAmdBlasTranspose trans, 
    size_t M, 
    size_t N, 
    size_t KL, 
    size_t KU, 
    cl_float2 alpha, 
    const cl_mem A, 
    size_t offa, 
    size_t lda, 
    const cl_mem X, 
    size_t offx, 
    int incx, 
    cl_float2 beta, 
    cl_mem Y, 
    size_t offy, 
    int incy, 
    cl_uint numCommandQueues, 
    cl_command_queue *commandQueues, 
    cl_uint numEventsInWaitList, 
    const cl_event *eventWaitList, 
    cl_event *events);


/**
 * @brief Matrix-vector product with a general rectangular banded matrix and
 * double-complex elements.
 *
 * Matrix-vector products:
 *   - \f$ Y \leftarrow \alpha A X + \beta Y \f$
 *   - \f$ Y \leftarrow \alpha A^T X + \beta Y \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] trans     How matrix \b A is to be transposed.
 * @param[in] M         Number of rows in banded matrix \b A.
 * @param[in] N         Number of columns in banded matrix \b A.
 * @param[in] KL        Number of sub-diagonals in banded matrix \b A.
 * @param[in] KU        Number of super-diagonals in banded matrix \b A.
 * @param[in] alpha     The factor of banded matrix \b A.
 * @param[in] A         Buffer object storing banded matrix \b A.
 * @param[in] offa      Offset in number of elements for the first element in banded matrix \b A.
 * @param[in] lda       Leading dimension of banded matrix \b A. It cannot be less
 *                      than ( \b KL + \b KU + 1 )
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of \b X. Must not be zero.
 * @param[in] beta      The factor of the vector \b Y.
 * @param[out] Y        Buffer object storing the vector \b y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of \b Y. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return The same result as the clAmdBlasDgbmv() function.
 *
 * @ingroup GBMV
 */
clAmdBlasStatus
clAmdBlasZgbmv(
    clAmdBlasOrder order,
    clAmdBlasTranspose trans,
    size_t M,
    size_t N,
    size_t KL,
    size_t KU,
    cl_double2 alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_double2 beta,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/*@}*/


/**
 * @defgroup TBMV TBMV  - Triangular Banded Matrix Vector multiply
 * @ingroup BLAS2
 */
/*@{*/

/**
 * @brief Matrix-vector product with a triangular banded matrix and
 * float elements.
 *
 * Matrix-vector products:
 *   - \f$ X \leftarrow  A X \f$
 *   - \f$ X \leftarrow  A^T X \f$
 *
 * @param[in] order				Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b A is to be transposed.
 * @param[in] diag				Specify whether matrix \b A is unit triangular.
 * @param[in] N					Number of rows/columns in banded matrix \b A.
 * @param[in] K					Number of sub-diagonals/super-diagonals in triangular banded matrix \b A.
 * @param[in] A					Buffer object storing matrix \b A.
 * @param[in] offa				Offset in number of elements for first element in matrix \b A.
 * @param[in] lda				Leading dimension of matrix \b A. It cannot be less
 *								than ( \b K + 1 )
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff		Temporary cl_mem scratch buffer object which can hold a
 *								minimum of (1 + (N-1)*abs(incx)) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - either \b N or \b incx is zero, or
 *     - K is greater than \b N - 1
 *     - the leading dimension is invalid;
 *   - \b clAmdBlasInvalidMemObject if either \b A or \b X object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup TBMV
 */
clAmdBlasStatus
clAmdBlasStbmv( 
    clAmdBlasOrder order, 
    clAmdBlasUplo uplo, 
    clAmdBlasTranspose trans, 
    clAmdBlasDiag diag, 
    size_t N, 
    size_t K, 
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/**
 * @example example_stbmv.c
 * Example of how to use the @ref clAmdBlasStbmv function.
 */


/**
 * @brief Matrix-vector product with a triangular banded matrix and
 * double elements.
 *
 * Matrix-vector products:
 *   - \f$ X \leftarrow  A X \f$
 *   - \f$ X \leftarrow  A^T X \f$
 *
 * @param[in] order				Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b A is to be transposed.
 * @param[in] diag				Specify whether matrix \b A is unit triangular.
 * @param[in] N					Number of rows/columns in banded matrix \b A.
 * @param[in] K					Number of sub-diagonals/super-diagonals in triangular banded matrix \b A.
 * @param[in] A					Buffer object storing matrix \b A.
 * @param[in] offa				Offset in number of elements for first element in matrix \b A.
 * @param[in] lda				Leading dimension of matrix \b A. It cannot be less
 *								than ( \b K + 1 )
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff		Temporary cl_mem scratch buffer object which can hold a
 *								minimum of (1 + (N-1)*abs(incx)) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasStbmv() function otherwise.
 *
 * @ingroup TBMV
 */
clAmdBlasStatus
clAmdBlasDtbmv(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose trans,
    clAmdBlasDiag diag,
    size_t N,
    size_t K,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);


/**
 * @brief Matrix-vector product with a triangular banded matrix and
 * float-complex elements.
 *
 * Matrix-vector products:
 *   - \f$ X \leftarrow  A X \f$
 *   - \f$ X \leftarrow  A^T X \f$
 *
 * @param[in] order				Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b A is to be transposed.
 * @param[in] diag				Specify whether matrix \b A is unit triangular.
 * @param[in] N					Number of rows/columns in banded matrix \b A.
 * @param[in] K					Number of sub-diagonals/super-diagonals in triangular banded matrix \b A.
 * @param[in] A					Buffer object storing matrix \b A.
 * @param[in] offa				Offset in number of elements for first element in matrix \b A.
 * @param[in] lda				Leading dimension of matrix \b A. It cannot be less
 *								than ( \b K + 1 )
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff		Temporary cl_mem scratch buffer object which can hold a
 *								minimum of (1 + (N-1)*abs(incx)) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
* @return The same result as the clAmdBlasStbmv() function.
 *
 * @ingroup TBMV
 */
clAmdBlasStatus
clAmdBlasCtbmv(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose trans,
    clAmdBlasDiag diag,
    size_t N,
    size_t K,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);


/**
 * @brief Matrix-vector product with a triangular banded matrix and
 * double-complex elements.
 *
 * Matrix-vector products:
 *   - \f$ X \leftarrow  A X \f$
 *   - \f$ X \leftarrow  A^T X \f$
 *
 * @param[in] order				Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b A is to be transposed.
 * @param[in] diag				Specify whether matrix \b A is unit triangular.
 * @param[in] N					Number of rows/columns in banded matrix \b A.
 * @param[in] K					Number of sub-diagonals/super-diagonals in triangular banded matrix \b A.
 * @param[in] A					Buffer object storing matrix \b A.
 * @param[in] offa				Offset in number of elements for first element in matrix \b A.
 * @param[in] lda				Leading dimension of matrix \b A. It cannot be less
 *								than ( \b K + 1 )
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] scratchBuff		Temporary cl_mem scratch buffer object which can hold a
 *								minimum of (1 + (N-1)*abs(incx)) elements
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
* @return The same result as the clAmdBlasDtbmv() function.
 *
 * @ingroup TBMV
 */
clAmdBlasStatus
clAmdBlasZtbmv(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose trans,
    clAmdBlasDiag diag,
    size_t N,
    size_t K,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/*@}*/


/**
 * @defgroup SBMV SBMV  - Symmetric Banded Matrix-Vector multiplication
 * @ingroup BLAS2
 */
/*@{*/

/**
 * @brief Matrix-vector product with a symmetric banded matrix and float elements.
 *
 * Matrix-vector products:
 * - \f$ Y \leftarrow \alpha A X + \beta Y \f$
 *
 * @param[in] order     Row/columns order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of rows and columns in banded matrix \b A.
 * @param[in] K			Number of sub-diagonals/super-diagonals in banded matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A			Buffer object storing matrix \b A.
 * @param[in] offa		Offset in number of elements for first element in matrix \b A.
 * @param[in] lda		Leading dimension of matrix \b A. It cannot be less
 *						than ( \b K + 1 )
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of vector \b X. It cannot be zero.
 * @param[in] beta      The factor of vector \b Y.
 * @param[out] Y        Buffer object storing vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of vector \b Y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - either \b N or \b incx is zero, or
 *     - K is greater than \b N - 1
 *     - the leading dimension is invalid;
 *   - \b clAmdBlasInvalidMemObject if either \b A or \b X object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup SBMV
 */
clAmdBlasStatus
clAmdBlasSsbmv(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    size_t N,
    size_t K,
    cl_float alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_float beta,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/**
 * @example example_ssbmv.c
 * This is an example of how to use the @ref clAmdBlasSsbmv function.
 */
 
 
/**
 * @brief Matrix-vector product with a symmetric banded matrix and double elements.
 *
 * Matrix-vector products:
 * - \f$ Y \leftarrow \alpha A X + \beta Y \f$
 *
 * @param[in] order     Row/columns order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of rows and columns in banded matrix \b A.
 * @param[in] K			Number of sub-diagonals/super-diagonals in banded matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A			Buffer object storing matrix \b A.
 * @param[in] offa		Offset in number of elements for first element in matrix \b A.
 * @param[in] lda		Leading dimension of matrix \b A. It cannot be less
 *						than ( \b K + 1 )
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of vector \b X. It cannot be zero.
 * @param[in] beta      The factor of vector \b Y.
 * @param[out] Y        Buffer object storing vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of vector \b Y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasSsbmv() function otherwise.
 *
 * @ingroup SBMV
 */
clAmdBlasStatus
clAmdBlasDsbmv(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    size_t N,
    size_t K,
    cl_double alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_double beta,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/*@}*/


/**
 * @defgroup HBMV HBMV  - Hermitian Banded Matrix-Vector multiplication
 * @ingroup BLAS2
 */
/*@{*/

/**
 * @brief Matrix-vector product with a hermitian banded matrix and float elements.
 *
 * Matrix-vector products:
 * - \f$ Y \leftarrow \alpha A X + \beta Y \f$
 *
 * @param[in] order     Row/columns order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of rows and columns in banded matrix \b A.
 * @param[in] K			Number of sub-diagonals/super-diagonals in banded matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A			Buffer object storing matrix \b A.
 * @param[in] offa		Offset in number of elements for first element in matrix \b A.
 * @param[in] lda		Leading dimension of matrix \b A. It cannot be less
 *						than ( \b K + 1 )
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of vector \b X. It cannot be zero.
 * @param[in] beta      The factor of vector \b Y.
 * @param[out] Y        Buffer object storing vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of vector \b Y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - either \b N or \b incx is zero, or
 *     - K is greater than \b N - 1
 *     - the leading dimension is invalid;
 *   - \b clAmdBlasInvalidMemObject if either \b A or \b X object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup HBMV
 */
clAmdBlasStatus
clAmdBlasChbmv(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    size_t N,
    size_t K,
    cl_float2 alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_float2 beta,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/**
 * @example example_chbmv.c
 * This is an example of how to use the @ref clAmdBlasChbmv function.
 */
 
 
/**
 * @brief Matrix-vector product with a hermitian banded matrix and double elements.
 *
 * Matrix-vector products:
 * - \f$ Y \leftarrow \alpha A X + \beta Y \f$
 *
 * @param[in] order     Row/columns order.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] N         Number of rows and columns in banded matrix \b A.
 * @param[in] K			Number of sub-diagonals/super-diagonals in banded matrix \b A.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A			Buffer object storing matrix \b A.
 * @param[in] offa		Offset in number of elements for first element in matrix \b A.
 * @param[in] lda		Leading dimension of matrix \b A. It cannot be less
 *						than ( \b K + 1 )
 * @param[in] X         Buffer object storing vector \b X.
 * @param[in] offx      Offset of first element of vector \b X in buffer object.
 *                      Counted in elements.
 * @param[in] incx      Increment for the elements of vector \b X. It cannot be zero.
 * @param[in] beta      The factor of vector \b Y.
 * @param[out] Y        Buffer object storing vector \b Y.
 * @param[in] offy      Offset of first element of vector \b Y in buffer object.
 *                      Counted in elements.
 * @param[in] incy      Increment for the elements of vector \b Y. It cannot be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasChbmv() function otherwise.
 *
 * @ingroup HBMV
 */
clAmdBlasStatus
clAmdBlasZhbmv(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    size_t N,
    size_t K,
    cl_double2 alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_double2 beta,
    cl_mem Y,
    size_t offy,
    int incy,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/*@}*/


/**
 * @defgroup TBSV TBSV  - Solving Triangular Banded matrix
 * @ingroup BLAS2
 */
/*@{*/

/**
 * @brief solving triangular banded matrix problems with float elements.
 *
 * Matrix-vector products:
 *   - \f$ A X \leftarrow  X \f$
 *   - \f$ A^T X \leftarrow  X \f$
 *
 * @param[in] order				Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b A is to be transposed.
 * @param[in] diag				Specify whether matrix \b A is unit triangular.
 * @param[in] N					Number of rows/columns in banded matrix \b A.
 * @param[in] K					Number of sub-diagonals/super-diagonals in triangular banded matrix \b A.
 * @param[in] A					Buffer object storing matrix \b A.
 * @param[in] offa				Offset in number of elements for first element in matrix \b A.
 * @param[in] lda				Leading dimension of matrix \b A. It cannot be less
 *								than ( \b K + 1 )
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - either \b N or \b incx is zero, or
 *     - K is greater than \b N - 1
 *     - the leading dimension is invalid;
 *   - \b clAmdBlasInvalidMemObject if either \b A or \b X object is
 *     Invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup TBSV
 */
 clAmdBlasStatus
clAmdBlasStbsv(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose trans,
    clAmdBlasDiag diag,
    size_t N,
    size_t K,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/**
 * @example example_stbsv.c
 * This is an example of how to use the @ref clAmdBlasStbsv function.
 */
 
 
/**
 * @brief solving triangular banded matrix problems with double elements.
 *
 * Matrix-vector products:
 *   - \f$ A X \leftarrow  X \f$
 *   - \f$ A^T X \leftarrow  X \f$
 *
 * @param[in] order				Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b A is to be transposed.
 * @param[in] diag				Specify whether matrix \b A is unit triangular.
 * @param[in] N					Number of rows/columns in banded matrix \b A.
 * @param[in] K					Number of sub-diagonals/super-diagonals in triangular banded matrix \b A.
 * @param[in] A					Buffer object storing matrix \b A.
 * @param[in] offa				Offset in number of elements for first element in matrix \b A.
 * @param[in] lda				Leading dimension of matrix \b A. It cannot be less
 *								than ( \b K + 1 )
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasStbsv() function otherwise.
 *
 * @ingroup TBSV
 */
clAmdBlasStatus
clAmdBlasDtbsv(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose trans,
    clAmdBlasDiag diag,
    size_t N,
    size_t K,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
    
/**
 * @brief solving triangular banded matrix problems with float-complex elements.
 *
 * Matrix-vector products:
 *   - \f$ A X \leftarrow  X \f$
 *   - \f$ A^T X \leftarrow  X \f$
 *
 * @param[in] order				Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b A is to be transposed.
 * @param[in] diag				Specify whether matrix \b A is unit triangular.
 * @param[in] N					Number of rows/columns in banded matrix \b A.
 * @param[in] K					Number of sub-diagonals/super-diagonals in triangular banded matrix \b A.
 * @param[in] A					Buffer object storing matrix \b A.
 * @param[in] offa				Offset in number of elements for first element in matrix \b A.
 * @param[in] lda				Leading dimension of matrix \b A. It cannot be less
 *								than ( \b K + 1 )
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return The same result as the clAmdBlasStbsv() function.
 *
 * @ingroup TBSV
 */
clAmdBlasStatus
clAmdBlasCtbsv(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose trans,
    clAmdBlasDiag diag,
    size_t N,
    size_t K,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
    
/**
 * @brief solving triangular banded matrix problems with double-complex elements.
 *
 * Matrix-vector products:
 *   - \f$ A X \leftarrow  X \f$
 *   - \f$ A^T X \leftarrow  X \f$
 *
 * @param[in] order				Row/column order.
 * @param[in] uplo				The triangle in matrix being referenced.
 * @param[in] trans				How matrix \b A is to be transposed.
 * @param[in] diag				Specify whether matrix \b A is unit triangular.
 * @param[in] N					Number of rows/columns in banded matrix \b A.
 * @param[in] K					Number of sub-diagonals/super-diagonals in triangular banded matrix \b A.
 * @param[in] A					Buffer object storing matrix \b A.
 * @param[in] offa				Offset in number of elements for first element in matrix \b A.
 * @param[in] lda				Leading dimension of matrix \b A. It cannot be less
 *								than ( \b K + 1 )
 * @param[out] X				Buffer object storing vector \b X.
 * @param[in] offx				Offset in number of elements for first element in vector \b X.
 * @param[in] incx				Increment for the elements of \b X. Must not be zero.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return The same result as the clAmdBlasDtbsv() function.
 *
 * @ingroup TBSV
 */
clAmdBlasStatus
clAmdBlasZtbsv(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose trans,
    clAmdBlasDiag diag,
    size_t N,
    size_t K,
    const cl_mem A,
    size_t offa,
    size_t lda,
    cl_mem X,
    size_t offx,
    int incx,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/*@}*/


/**
 * @defgroup BLAS3 BLAS-3 functions
 *
 * The Level 3 Basic Linear Algebra Subprograms are funcions that perform
 * matrix-matrix operations.
 */
/*@{*/
/*@}*/

/**
 * @defgroup GEMM GEMM - GEneral Matrix-matrix Multiplication
 * @ingroup BLAS3
 */
/*@{*/

/**
 * @brief Matrix-matrix product of general rectangular matrices with float
 * elements.
 *
 * Matrix-matrix products:
 *   - \f$ C \leftarrow \alpha A B + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B + \beta C \f$
 *   - \f$ C \leftarrow \alpha A B^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B^T + \beta C \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] transB    How matrix \b B is to be transposed.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] K         Number of columns in matrix \b A and rows in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b K when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M when the
 *                      parameter is set to \b clAmdBlasColumnMajor.
 * @param[in] B         Buffer object storing matrix \b B.
 * @param[in] ldb       Leading dimension of matrix \b B. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b K
 *                      when it is set to \b clAmdBlasColumnMajor.
 * @param[in] beta      The factor of matrix \b C.
 * @param[out] C        Buffer object storing matrix \b C.
 * @param[in] ldc       Leading dimension of matrix \b C. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M when
 *                      it is set to \b clAmdBlasColumnMajorOrder.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasSgemmEx() instead.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - \b M, \b N or \b K is zero, or
 *     - any of the leading dimensions is invalid;
 *     - the matrix sizes lead to accessing outsize of any of the buffers;
 *   - \b clAmdBlasInvalidMemObject if A, B, or C object is invalid,
 *     or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfResources if you use image-based function implementation
 *     and no suitable scratch image available;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs to
 *     was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup GEMM
 */
clAmdBlasStatus
clAmdBlasSgemm(
    clAmdBlasOrder order,
    clAmdBlasTranspose transA,
    clAmdBlasTranspose transB,
    size_t M,
    size_t N,
    size_t K,
    cl_float alpha,
    const cl_mem A,
    size_t lda,
    const cl_mem B,
    size_t ldb,
    cl_float beta,
    cl_mem C,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @example example_sgemm.c
 * This is an example of how to use the @ref clAmdBlasSgemm function.
 */

/**
 * @brief Matrix-matrix product of general rectangular matrices with double
 * elements.
 *
 * Matrix-matrix products:
 *   - \f$ C \leftarrow \alpha A B + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B + \beta C \f$
 *   - \f$ C \leftarrow \alpha A B^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B^T + \beta C \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] transB    How matrix \b B is to be transposed.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] K         Number of columns in matrix \b A and rows in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. For detailed description,
 *                      see clAmdBlasSgemm().
 * @param[in] B         Buffer object storing matrix \b B.
 * @param[in] ldb       Leading dimension of matrix \b B. For detailed description,
 *                      see clAmdBlasSgemm().
 * @param[in] beta      The factor of matrix \b C.
 * @param[out] C        Buffer object storing matrix \b C.
 * @param[in] ldc       Leading dimension of matrix \b C. For detailed description,
 *                      see clAmdBlasSgemm().
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasDgemmEx() instead.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasSgemm() function otherwise.
 *
 * @ingroup GEMM
 */
clAmdBlasStatus
clAmdBlasDgemm(
    clAmdBlasOrder order,
    clAmdBlasTranspose transA,
    clAmdBlasTranspose transB,
    size_t M,
    size_t N,
    size_t K,
    cl_double alpha,
    const cl_mem A,
    size_t lda,
    const cl_mem B,
    size_t ldb,
    cl_double beta,
    cl_mem C,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Matrix-matrix product of general rectangular matrices with float
 * complex elements.
 *
 * Matrix-matrix products:
 *   - \f$ C \leftarrow \alpha A B + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B + \beta C \f$
 *   - \f$ C \leftarrow \alpha A B^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B^T + \beta C \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] transB    How matrix \b B is to be transposed.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] K         Number of columns in matrix \b A and rows in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. For detailed description,
 *                      see clAmdBlasSgemm().
 * @param[in] B         Buffer object storing matrix \b B.
 * @param[in] ldb       Leading dimension of matrix \b B. For detailed description,
 *                      see clAmdBlasSgemm().
 * @param[in] beta      The factor of matrix \b C.
 * @param[out] C        Buffer object storing matrix \b C.
 * @param[in] ldc       Leading dimension of matrix \b C. For detailed description,
 *                      see clAmdBlasSgemm().
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasCgemmEx() instead.
 *
 * @return The same result as the clAmdBlasSgemm() function.
 *
 * @ingroup GEMM
 */
clAmdBlasStatus
clAmdBlasCgemm(
    clAmdBlasOrder order,
    clAmdBlasTranspose transA,
    clAmdBlasTranspose transB,
    size_t M,
    size_t N,
    size_t K,
    FloatComplex alpha,
    const cl_mem A,
    size_t lda,
    const cl_mem B,
    size_t ldb,
    FloatComplex beta,
    cl_mem C,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Matrix-matrix product of general rectangular matrices with double
 * complex elements.
 *
 * Matrix-matrix products:
 *   - \f$ C \leftarrow \alpha A B + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B + \beta C \f$
 *   - \f$ C \leftarrow \alpha A B^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B^T + \beta C \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] transB    How matrix \b B is to be transposed.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] K         Number of columns in matrix \b A and rows in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. For detailed description,
 *                      see clAmdBlasSgemm().
 * @param[in] B         Buffer object storing matrix \b B.
 * @param[in] ldb       Leading dimension of matrix \b B. For detailed description,
 *                      see clAmdBlasSgemm().
 * @param[in] beta      The factor of matrix \b C.
 * @param[out] C        Buffer object storing matrix \b C.
 * @param[in] ldc       Leading dimension of matrix \b C. For detailed description,
 *                      see clAmdBlasSgemm().
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasZgemmEx() instead.
 *
 * @return The same result as the clAmdBlasDgemm() function.
 *
 * @ingroup GEMM
 */
clAmdBlasStatus
clAmdBlasZgemm(
    clAmdBlasOrder order,
    clAmdBlasTranspose transA,
    clAmdBlasTranspose transB,
    size_t M,
    size_t N,
    size_t K,
    DoubleComplex alpha,
    const cl_mem A,
    size_t lda,
    const cl_mem B,
    size_t ldb,
    DoubleComplex beta,
    cl_mem C,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Matrix-matrix product of general rectangular matrices with float
 *        elements. Extended version.
 *
 * Matrix-matrix products:
 *   - \f$ C \leftarrow \alpha A B + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B + \beta C \f$
 *   - \f$ C \leftarrow \alpha A B^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B^T + \beta C \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] transB    How matrix \b B is to be transposed.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] K         Number of columns in matrix \b A and rows in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b K when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M when the
 *                      parameter is set to \b clAmdBlasColumnMajor.
 * @param[in] B         Buffer object storing matrix \b B.
 * @param[in] offB      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b K
 *                      when it is set to \b clAmdBlasColumnMajor.
 * @param[in] beta      The factor of matrix \b C.
 * @param[out] C        Buffer object storing matrix \b C.
 * @param[in]  offC     Offset of the first element of the matrix \b C in the
 *                      buffer object. Counted in elements.
 * @param[in] ldc       Leading dimension of matrix \b C. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M when
 *                      it is set to \b clAmdBlasColumnMajorOrder.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidValue if either \b offA, \b offB or \b offC exceeds
 *        the size of the respective buffer object;
 *   - the same error codes as clAmdBlasSgemm() otherwise.
 *
 * @ingroup GEMM
 */
clAmdBlasStatus
clAmdBlasSgemmEx(
    clAmdBlasOrder order,
    clAmdBlasTranspose transA,
    clAmdBlasTranspose transB,
    size_t M,
    size_t N,
    size_t K,
    cl_float alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem B,
    size_t offB,
    size_t ldb,
    cl_float beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @example example_sgemm.c
 * This is an example of how to use the @ref clAmdBlasSgemmEx function.
 */

/**
 * @brief Matrix-matrix product of general rectangular matrices with double
 *        elements. Extended version.
 *
 * Matrix-matrix products:
 *   - \f$ C \leftarrow \alpha A B + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B + \beta C \f$
 *   - \f$ C \leftarrow \alpha A B^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B^T + \beta C \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] transB    How matrix \b B is to be transposed.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] K         Number of columns in matrix \b A and rows in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. For detailed description,
 *                      see clAmdBlasSgemm().
 * @param[in] B         Buffer object storing matrix \b B.
 * @param[in] offB      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. For detailed description,
 *                      see clAmdBlasSgemm().
 * @param[in] beta      The factor of matrix \b C.
 * @param[out] C        Buffer object storing matrix \b C.
 * @param[in] offC      Offset of the first element of the matrix \b C in the
 *                      buffer object. Counted in elements.
 * @param[in] ldc       Leading dimension of matrix \b C. For detailed description,
 *                      see clAmdBlasSgemm().
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *        point arithmetic with double precision;
 *   - \b clAmdBlasInvalidValue if either \b offA, \b offB or offC exceeds
 *        the size of the respective buffer object;
 *   - the same error codes as the clAmdBlasSgemm() function otherwise.
 *
 * @ingroup GEMM
 */
clAmdBlasStatus
clAmdBlasDgemmEx(
    clAmdBlasOrder order,
    clAmdBlasTranspose transA,
    clAmdBlasTranspose transB,
    size_t M,
    size_t N,
    size_t K,
    cl_double alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem B,
    size_t offB,
    size_t ldb,
    cl_double beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Matrix-matrix product of general rectangular matrices with float
 *        complex elements. Extended version.
 *
 * Matrix-matrix products:
 *   - \f$ C \leftarrow \alpha A B + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B + \beta C \f$
 *   - \f$ C \leftarrow \alpha A B^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B^T + \beta C \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] transB    How matrix \b B is to be transposed.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] K         Number of columns in matrix \b A and rows in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. For detailed description,
 *                      see clAmdBlasSgemm().
 * @param[in] B         Buffer object storing matrix \b B.
 * @param[in] offB      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. For detailed description,
 *                      see clAmdBlasSgemm().
 * @param[in] beta      The factor of matrix \b C.
 * @param[out] C        Buffer object storing matrix \b C.
 * @param[in] offC      Offset of the first element of the matrix \b C in the
 *                      buffer object. Counted in elements.
 * @param[in] ldc       Leading dimension of matrix \b C. For detailed description,
 *                      see clAmdBlasSgemm().
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidValue if either \b offA, \b offB or offC exceeds
 *        the size of the respective buffer object;
 *   - the same error codes as the clAmdBlasSgemm() function otherwise.
 *
 * @ingroup GEMM
 */
clAmdBlasStatus
clAmdBlasCgemmEx(
    clAmdBlasOrder order,
    clAmdBlasTranspose transA,
    clAmdBlasTranspose transB,
    size_t M,
    size_t N,
    size_t K,
    FloatComplex alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem B,
    size_t offB,
    size_t ldb,
    FloatComplex beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Matrix-matrix product of general rectangular matrices with double
 *        complex elements. Exteneded version.
 *
 * Matrix-matrix products:
 *   - \f$ C \leftarrow \alpha A B + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B + \beta C \f$
 *   - \f$ C \leftarrow \alpha A B^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B^T + \beta C \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] transB    How matrix \b B is to be transposed.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] K         Number of columns in matrix \b A and rows in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. For detailed description,
 *                      see clAmdBlasSgemm().
 * @param[in] B         Buffer object storing matrix \b B.
 * @param[in] offB      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. For detailed description,
 *                      see clAmdBlasSgemm().
 * @param[in] beta      The factor of matrix \b C.
 * @param[out] C        Buffer object storing matrix \b C.
 * @param[in] offC      Offset of the first element of the matrix \b C in the
 *                      buffer object. Counted in elements.
 * @param[in] ldc       Leading dimension of matrix \b C. For detailed description,
 *                      see clAmdBlasSgemm().
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *        point arithmetic with double precision;
 *   - \b clAmdBlasInvalidValue if either \b offA, \b offB or offC exceeds
 *        the size of the respective buffer object;
 *   - the same error codes as the clAmdBlasSgemm() function otherwise.
 *
 * @ingroup GEMM
 */
clAmdBlasStatus
clAmdBlasZgemmEx(
    clAmdBlasOrder order,
    clAmdBlasTranspose transA,
    clAmdBlasTranspose transB,
    size_t M,
    size_t N,
    size_t K,
    DoubleComplex alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem B,
    size_t offB,
    size_t ldb,
    DoubleComplex beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/*@}*/

/**
 * @defgroup TRMM TRMM - Triangular Matrix-matrix Multiplication
 * @ingroup BLAS3
 */
/*@{*/

/**
 * @brief Multiplying a matrix by a triangular matrix with float elements.
 *
 * Matrix-triangular matrix products:
 *   - \f$ B \leftarrow \alpha T B \f$
 *   - \f$ B \leftarrow \alpha T^T B \f$
 *   - \f$ B \leftarrow \alpha B T \f$
 *   - \f$ B \leftarrow \alpha B T^T \f$
 *
 * where \b T is an upper or lower triangular matrix.
 *
 * @param[in] order     Row/column order.
 * @param[in] side      The side of triangular matrix.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] diag      Specify whether matrix is unit triangular.
 * @param[in] M         Number of rows in matrices \b A and \b B.
 * @param[in] N         Number of columns in matrices \b A and \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b M when the \b side parameter is set to
 *                      \b clAmdBlasLeft,\n or less than \b N when it is set
 *                      to \b clAmdBlasRight.
 * @param[out] B        Buffer object storing matrix \b B.
 * @param[in] ldb       Leading dimension of matrix \b B. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or not less than \b M
 *                      when it is set to \b clAmdBlasColumnMajor.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasStrmmEx() instead.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - \b M, \b N, or \b K is zero, or
 *     - any of the leading dimensions is invalid;
 *     - the matrix sizes lead to accessing outsize of any of the buffers;
 *   - \b clAmdBlasInvalidMemObject if A, B, or C object is invalid,
 *     or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfResources if you use image-based function implementation
 *     and no suitable scratch image available;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs to
 *     was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup TRMM
 */
clAmdBlasStatus
clAmdBlasStrmm(
    clAmdBlasOrder order,
    clAmdBlasSide side,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    clAmdBlasDiag diag,
    size_t M,
    size_t N,
    cl_float alpha,
    const cl_mem A,
    size_t lda,
    cl_mem B,
    size_t ldb,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @example example_strmm.c
 * This is an example of how to use the @ref clAmdBlasStrmm function.
 */

/**
 * @brief Multiplying a matrix by a triangular matrix with double elements.
 *
 * Matrix-triangular matrix products:
 *   - \f$ B \leftarrow \alpha T B \f$
 *   - \f$ B \leftarrow \alpha T^T B \f$
 *   - \f$ B \leftarrow \alpha B T \f$
 *   - \f$ B \leftarrow \alpha B T^T \f$
 *
 * where \b T is an upper or lower triangular matrix.
 *
 * @param[in] order     Row/column order.
 * @param[in] side      The side of triangular matrix.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] diag      Specify whether matrix is unit triangular.
 * @param[in] M         Number of rows in matrices \b A and \b B.
 * @param[in] N         Number of columns in matrices \b A and \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. For detailed
 *                      description, see clAmdBlasStrmm().
 * @param[out] B        Buffer object storing matrix \b B.
 * @param[in] ldb       Leading dimension of matrix \b B. For detailed
 *                      description, see clAmdBlasStrmm().
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasDtrmmEx() instead.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasStrmm() function otherwise.
 *
 * @ingroup TRMM
 */
clAmdBlasStatus
clAmdBlasDtrmm(
    clAmdBlasOrder order,
    clAmdBlasSide side,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    clAmdBlasDiag diag,
    size_t M,
    size_t N,
    cl_double alpha,
    const cl_mem A,
    size_t lda,
    cl_mem B,
    size_t ldb,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Multiplying a matrix by a triangular matrix with float complex
 * elements.
 *
 * Matrix-triangular matrix products:
 *   - \f$ B \leftarrow \alpha T B \f$
 *   - \f$ B \leftarrow \alpha T^T B \f$
 *   - \f$ B \leftarrow \alpha B T \f$
 *   - \f$ B \leftarrow \alpha B T^T \f$
 *
 * where \b T is an upper or lower triangular matrix.
 * @param[in] order     Row/column order.
 * @param[in] side      The side of triangular matrix.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] diag      Specify whether matrix is unit triangular.
 * @param[in] M         Number of rows in matrices \b A and \b B.
 * @param[in] N         Number of columns in matrices \b A and \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. For detailed
 *                      description, see clAmdBlasStrmm().
 * @param[out] B        Buffer object storing matrix \b B.
 * @param[in] ldb       Leading dimension of matrix \b B. For detailed
 *                      description, see clAmdBlasStrmm().
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasCtrmmEx() instead.
 *
 * @return The same result as the clAmdBlasStrmm() function.
 *
 * @ingroup TRMM
 */
clAmdBlasStatus
clAmdBlasCtrmm(
    clAmdBlasOrder order,
    clAmdBlasSide side,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    clAmdBlasDiag diag,
    size_t M,
    size_t N,
    FloatComplex alpha,
    const cl_mem A,
    size_t lda,
    cl_mem B,
    size_t ldb,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Multiplying a matrix by a triangular matrix with double complex
 * elements.
 *
 * Matrix-triangular matrix products:
 *   - \f$ B \leftarrow \alpha T B \f$
 *   - \f$ B \leftarrow \alpha T^T B \f$
 *   - \f$ B \leftarrow \alpha B T \f$
 *   - \f$ B \leftarrow \alpha B T^T \f$
 *
 * where \b T is an upper or lower triangular matrix.
 *
 * @param[in] order     Row/column order.
 * @param[in] side      The side of triangular matrix.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] diag      Specify whether matrix is unit triangular.
 * @param[in] M         Number of rows in matrices \b A and \b B.
 * @param[in] N         Number of columns in matrices \b A and \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. For detailed
 *                      description, see clAmdBlasStrmm().
 * @param[out] B        Buffer object storing matrix \b B.
 * @param[in] ldb       Leading dimension of matrix \b B. For detailed
 *                      description, see clAmdBlasStrmm().
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasZtrmmEx() instead.
 *
 * @return The same result as the clAmdBlasDtrmm() function.
 *
 * @ingroup TRMM
 */
clAmdBlasStatus
clAmdBlasZtrmm(
    clAmdBlasOrder order,
    clAmdBlasSide side,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    clAmdBlasDiag diag,
    size_t M,
    size_t N,
    DoubleComplex alpha,
    const cl_mem A,
    size_t lda,
    cl_mem B,
    size_t ldb,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Multiplying a matrix by a triangular matrix with float elements.
 *        Extended version.
 *
 * Matrix-triangular matrix products:
 *   - \f$ B \leftarrow \alpha T B \f$
 *   - \f$ B \leftarrow \alpha T^T B \f$
 *   - \f$ B \leftarrow \alpha B T \f$
 *   - \f$ B \leftarrow \alpha B T^T \f$
 *
 * where \b T is an upper or lower triangular matrix.
 *
 * @param[in] order     Row/column order.
 * @param[in] side      The side of triangular matrix.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] diag      Specify whether matrix is unit triangular.
 * @param[in] M         Number of rows in matrices \b A and \b B.
 * @param[in] N         Number of columns in matrices \b A and \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b M when the \b side parameter is set to
 *                      \b clAmdBlasLeft,\n or less than \b N when it is set
 *                      to \b clAmdBlasRight.
 * @param[out] B        Buffer object storing matrix \b B.
 * @param[in] offB      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or not less than \b M
 *                      when it is set to \b clAmdBlasColumnMajor.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidValue if either \b offA or \b offB exceeds the size
 *        of the respective buffer object;
 *   - the same error codes as clAmdBlasStrmm() otherwise.
 *
 * @ingroup TRMM
 */
clAmdBlasStatus
clAmdBlasStrmmEx(
    clAmdBlasOrder order,
    clAmdBlasSide side,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    clAmdBlasDiag diag,
    size_t M,
    size_t N,
    cl_float alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    cl_mem B,
    size_t offB,
    size_t ldb,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @example example_strmm.c
 * This is an example of how to use the @ref clAmdBlasStrmmEx function.
 */

/**
 * @brief Multiplying a matrix by a triangular matrix with double elements.
 *        Extended version.
 *
 * Matrix-triangular matrix products:
 *   - \f$ B \leftarrow \alpha T B \f$
 *   - \f$ B \leftarrow \alpha T^T B \f$
 *   - \f$ B \leftarrow \alpha B T \f$
 *   - \f$ B \leftarrow \alpha B T^T \f$
 *
 * where \b T is an upper or lower triangular matrix.
 *
 * @param[in] order     Row/column order.
 * @param[in] side      The side of triangular matrix.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] diag      Specify whether matrix is unit triangular.
 * @param[in] M         Number of rows in matrices \b A and \b B.
 * @param[in] N         Number of columns in matrices \b A and \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. For detailed
 *                      description, see clAmdBlasStrmm().
 * @param[out] B        Buffer object storing matrix \b B.
 * @param[in] offB      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. For detailed
 *                      description, see clAmdBlasStrmm().
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - \b clAmdBlasInvalidValue if either \b offA or \b offB exceeds the size
 *        of the respective buffer object;
 *   - the same error codes as the clAmdBlasStrmm() function otherwise.
 *
 * @ingroup TRMM
 */
clAmdBlasStatus
clAmdBlasDtrmmEx(
    clAmdBlasOrder order,
    clAmdBlasSide side,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    clAmdBlasDiag diag,
    size_t M,
    size_t N,
    cl_double alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    cl_mem B,
    size_t offB,
    size_t ldb,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Multiplying a matrix by a triangular matrix with float complex
 *        elements. Extended version.
 *
 * Matrix-triangular matrix products:
 *   - \f$ B \leftarrow \alpha T B \f$
 *   - \f$ B \leftarrow \alpha T^T B \f$
 *   - \f$ B \leftarrow \alpha B T \f$
 *   - \f$ B \leftarrow \alpha B T^T \f$
 *
 * where \b T is an upper or lower triangular matrix.
 * @param[in] order     Row/column order.
 * @param[in] side      The side of triangular matrix.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] diag      Specify whether matrix is unit triangular.
 * @param[in] M         Number of rows in matrices \b A and \b B.
 * @param[in] N         Number of columns in matrices \b A and \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. For detailed
 *                      description, see clAmdBlasStrmm().
 * @param[out] B        Buffer object storing matrix \b B.
 * @param[in] offB      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. For detailed
 *                      description, see clAmdBlasStrmm().
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidValue if either \b offA or \b offB exceeds the size
 *        of the respective buffer object;
 *   - the same error codes as clAmdBlasStrmm() otherwise.
 *
 * @ingroup TRMM
 */
clAmdBlasStatus
clAmdBlasCtrmmEx(
    clAmdBlasOrder order,
    clAmdBlasSide side,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    clAmdBlasDiag diag,
    size_t M,
    size_t N,
    FloatComplex alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    cl_mem B,
    size_t offB,
    size_t ldb,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Multiplying a matrix by a triangular matrix with double complex
 *        elements. Extended version.
 *
 * Matrix-triangular matrix products:
 *   - \f$ B \leftarrow \alpha T B \f$
 *   - \f$ B \leftarrow \alpha T^T B \f$
 *   - \f$ B \leftarrow \alpha B T \f$
 *   - \f$ B \leftarrow \alpha B T^T \f$
 *
 * where \b T is an upper or lower triangular matrix.
 *
 * @param[in] order     Row/column order.
 * @param[in] side      The side of triangular matrix.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] diag      Specify whether matrix is unit triangular.
 * @param[in] M         Number of rows in matrices \b A and \b B.
 * @param[in] N         Number of columns in matrices \b A and \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. For detailed
 *                      description, see clAmdBlasStrmm().
 * @param[out] B        Buffer object storing matrix \b B.
 * @param[in] offB      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. For detailed
 *                      description, see clAmdBlasStrmm().
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - \b clAmdBlasInvalidValue if either \b offA or \b offB exceeds the size
 *        of the respective buffer object;
 *   - the same error codes as the clAmdBlasStrmm() function otherwise.
 *
 * @ingroup TRMM
 */
clAmdBlasStatus
clAmdBlasZtrmmEx(
    clAmdBlasOrder order,
    clAmdBlasSide side,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    clAmdBlasDiag diag,
    size_t M,
    size_t N,
    DoubleComplex alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    cl_mem B,
    size_t offB,
    size_t ldb,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/*@}*/

/**
 * @defgroup TRSM TRSM - Solving triangular systems of equations
 * @ingroup BLAS3
 */
/*@{*/

/**
 * @brief Solving triangular systems of equations with multiple right-hand
 * sides and float elements.
 *
 * Solving triangular systems of equations:
 *   - \f$ B \leftarrow \alpha T^{-1} B \f$
 *   - \f$ B \leftarrow \alpha T^{-T} B \f$
 *   - \f$ B \leftarrow \alpha B T^{-1} \f$
 *   - \f$ B \leftarrow \alpha B T^{-T} \f$
 *
 * where \b T is an upper or lower triangular matrix.
 *
 * @param[in] order     Row/column order.
 * @param[in] side      The side of triangular matrix.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] diag      Specify whether matrix is unit triangular.
 * @param[in] M         Number of rows in matrices \b A and \b B.
 * @param[in] N         Number of columns in matrices \b A and \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b N when the \b side parameter is set to
 *                      \b clAmdBlasRowLeft,\n or less than \b M
 *                      when it is set to \b clAmdBlasRight.
 * @param[out] B        Buffer object storing matrix \b B.
 * @param[in] ldb       Leading dimension of matrix \b B. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M
 *                      when it is set to \b clAmdBlasColumnMajor.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasStrsmEx() instead.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - \b M, \b N or \b K is zero, or
 *     - any of the leading dimensions is invalid;
 *     - the matrix sizes lead to accessing outsize of any of the buffers;
 *   - \b clAmdBlasInvalidMemObject if A, B, or C object is invalid,
 *     or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfResources if you use image-based function implementation
 *     and no suitable scratch image available;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs
 *     to was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup TRSM
 */
clAmdBlasStatus
clAmdBlasStrsm(
    clAmdBlasOrder order,
    clAmdBlasSide side,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    clAmdBlasDiag diag,
    size_t M,
    size_t N,
    cl_float alpha,
    const cl_mem A,
    size_t lda,
    cl_mem B,
    size_t ldb,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @example example_strsm.c
 * This is an example of how to use the @ref clAmdBlasStrsm function.
 */

/**
 * @brief Solving triangular systems of equations with multiple right-hand
 * sides and double elements.
 *
 * Solving triangular systems of equations:
 *   - \f$ B \leftarrow \alpha T^{-1} B \f$
 *   - \f$ B \leftarrow \alpha T^{-T} B \f$
 *   - \f$ B \leftarrow \alpha B T^{-1} \f$
 *   - \f$ B \leftarrow \alpha B T^{-T} \f$
 *
 * where \b T is an upper or lower triangular matrix.
 *
 * @param[in] order     Row/column order.
 * @param[in] side      The side of triangular matrix.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] diag      Specify whether matrix is unit triangular.
 * @param[in] M         Number of rows in matrices \b A and \b B.
 * @param[in] N         Number of columns in matrices \b A and \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. For detailed
 *                      description, see clAmdBlasStrsm().
 * @param[out] B        Buffer object storing matrix \b B.
 * @param[in] ldb       Leading dimension of matrix \b B. For detailed
 *                      description, see clAmdBlasStrsm().
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasDtrsmEx() instead.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasStrsm() function otherwise.
 *
 * @ingroup TRSM
 */
clAmdBlasStatus
clAmdBlasDtrsm(
    clAmdBlasOrder order,
    clAmdBlasSide side,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    clAmdBlasDiag diag,
    size_t M,
    size_t N,
    cl_double alpha,
    const cl_mem A,
    size_t lda,
    cl_mem B,
    size_t ldb,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Solving triangular systems of equations with multiple right-hand
 * sides and float complex elements.
 *
 * Solving triangular systems of equations:
 *   - \f$ B \leftarrow \alpha T^{-1} B \f$
 *   - \f$ B \leftarrow \alpha T^{-T} B \f$
 *   - \f$ B \leftarrow \alpha B T^{-1} \f$
 *   - \f$ B \leftarrow \alpha B T^{-T} \f$
 *
 * where \b T is an upper or lower triangular matrix.
 *
 * @param[in] order     Row/column order.
 * @param[in] side      The side of triangular matrix.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] diag      Specify whether matrix is unit triangular.
 * @param[in] M         Number of rows in matrices \b A and \b B.
 * @param[in] N         Number of columns in matrices \b A and \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. For detailed
 *                      description, see clAmdBlasStrsm().
 * @param[out] B        Buffer object storing matrix \b B.
 * @param[in] ldb       Leading dimension of matrix \b B. For detailed
 *                      description, see clAmdBlasStrsm().
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasCtrsmEx() instead.
 *
 * @return The same result as the clAmdBlasStrsm() function.
 *
 * @ingroup TRSM
 */
clAmdBlasStatus
clAmdBlasCtrsm(
    clAmdBlasOrder order,
    clAmdBlasSide side,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    clAmdBlasDiag diag,
    size_t M,
    size_t N,
    FloatComplex alpha,
    const cl_mem A,
    size_t lda,
    cl_mem B,
    size_t ldb,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Solving triangular systems of equations with multiple right-hand
 * sides and double complex elements.
 *
 * Solving triangular systems of equations:
 *   - \f$ B \leftarrow \alpha T^{-1} B \f$
 *   - \f$ B \leftarrow \alpha T^{-T} B \f$
 *   - \f$ B \leftarrow \alpha B T^{-1} \f$
 *   - \f$ B \leftarrow \alpha B T^{-T} \f$
 *
 * where \b T is an upper or lower triangular matrix.
 *
 * @param[in] order     Row/column order.
 * @param[in] side      The side of triangular matrix.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] diag      Specify whether matrix is unit triangular.
 * @param[in] M         Number of rows in matrices \b A and \b B.
 * @param[in] N         Number of columns in matrices \b A and \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] lda       Leading dimension of matrix \b A. For detailed
 *                      description, see clAmdBlasStrsm().
 * @param[out] B        Buffer object storing matrix \b B.
 * @param[in] ldb       Leading dimension of matrix \b B. For detailed
 *                      description, see clAmdBlasStrsm().
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasZtrsmEx() instead.
 *
 * @return The same result as the clAmdBlasDtrsm() function.
 *
 * @ingroup TRSM
 */
clAmdBlasStatus
clAmdBlasZtrsm(
    clAmdBlasOrder order,
    clAmdBlasSide side,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    clAmdBlasDiag diag,
    size_t M,
    size_t N,
    DoubleComplex alpha,
    const cl_mem A,
    size_t lda,
    cl_mem B,
    size_t ldb,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Solving triangular systems of equations with multiple right-hand
 *        sides and float elements. Extended version.
 *
 * Solving triangular systems of equations:
 *   - \f$ B \leftarrow \alpha T^{-1} B \f$
 *   - \f$ B \leftarrow \alpha T^{-T} B \f$
 *   - \f$ B \leftarrow \alpha B T^{-1} \f$
 *   - \f$ B \leftarrow \alpha B T^{-T} \f$
 *
 * where \b T is an upper or lower triangular matrix.
 *
 * @param[in] order     Row/column order.
 * @param[in] side      The side of triangular matrix.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] diag      Specify whether matrix is unit triangular.
 * @param[in] M         Number of rows in matrices \b A and \b B.
 * @param[in] N         Number of columns in matrices \b A and \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b N when the \b side parameter is set to
 *                      \b clAmdBlasRowLeft,\n or less than \b M
 *                      when it is set to \b clAmdBlasRight.
 * @param[out] B        Buffer object storing matrix \b B.
 * @param[in] offB      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M
 *                      when it is set to \b clAmdBlasColumnMajor.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidValue if either \b offA or \b offB exceeds the size
 *        of the respective buffer object;
 *   - the same error codes as clAmdBlasStrsm() otherwise.
 *
 * @ingroup TRSM
 */
clAmdBlasStatus
clAmdBlasStrsmEx(
    clAmdBlasOrder order,
    clAmdBlasSide side,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    clAmdBlasDiag diag,
    size_t M,
    size_t N,
    cl_float alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    cl_mem B,
    size_t offB,
    size_t ldb,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @example example_strsm.c
 * This is an example of how to use the @ref clAmdBlasStrsmEx function.
 */

/**
 * @brief Solving triangular systems of equations with multiple right-hand
 *        sides and double elements. Extended version.
 *
 * Solving triangular systems of equations:
 *   - \f$ B \leftarrow \alpha T^{-1} B \f$
 *   - \f$ B \leftarrow \alpha T^{-T} B \f$
 *   - \f$ B \leftarrow \alpha B T^{-1} \f$
 *   - \f$ B \leftarrow \alpha B T^{-T} \f$
 *
 * where \b T is an upper or lower triangular matrix.
 *
 * @param[in] order     Row/column order.
 * @param[in] side      The side of triangular matrix.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] diag      Specify whether matrix is unit triangular.
 * @param[in] M         Number of rows in matrices \b A and \b B.
 * @param[in] N         Number of columns in matrices \b A and \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. For detailed
 *                      description, see clAmdBlasStrsm().
 * @param[out] B        Buffer object storing matrix \b B.
 * @param[in] offB      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. For detailed
 *                      description, see clAmdBlasStrsm().
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *        point arithmetic with double precision;
 *   - \b clAmdBlasInvalidValue if either \b offA or \b offB exceeds the size
 *        of the respective buffer object;
 *   - the same error codes as the clAmdBlasStrsm() function otherwise.
 *
 * @ingroup TRSM
 */
clAmdBlasStatus
clAmdBlasDtrsmEx(
    clAmdBlasOrder order,
    clAmdBlasSide side,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    clAmdBlasDiag diag,
    size_t M,
    size_t N,
    cl_double alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    cl_mem B,
    size_t offB,
    size_t ldb,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Solving triangular systems of equations with multiple right-hand
 *        sides and float complex elements. Extended version.
 *
 * Solving triangular systems of equations:
 *   - \f$ B \leftarrow \alpha T^{-1} B \f$
 *   - \f$ B \leftarrow \alpha T^{-T} B \f$
 *   - \f$ B \leftarrow \alpha B T^{-1} \f$
 *   - \f$ B \leftarrow \alpha B T^{-T} \f$
 *
 * where \b T is an upper or lower triangular matrix.
 *
 * @param[in] order     Row/column order.
 * @param[in] side      The side of triangular matrix.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] diag      Specify whether matrix is unit triangular.
 * @param[in] M         Number of rows in matrices \b A and \b B.
 * @param[in] N         Number of columns in matrices \b A and \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. For detailed
 *                      description, see clAmdBlasStrsm().
 * @param[out] B        Buffer object storing matrix \b B.
 * @param[in] offB      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. For detailed
 *                      description, see clAmdBlasStrsm().
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidValue if either \b offA or \b offB exceeds the size
 *        of the respective buffer object;
 *   - the same error codes as clAmdBlasStrsm() otherwise.
 *
 * @ingroup TRSM
 */
clAmdBlasStatus
clAmdBlasCtrsmEx(
    clAmdBlasOrder order,
    clAmdBlasSide side,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    clAmdBlasDiag diag,
    size_t M,
    size_t N,
    FloatComplex alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    cl_mem B,
    size_t offB,
    size_t ldb,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Solving triangular systems of equations with multiple right-hand
 *        sides and double complex elements. Extended version.
 *
 * Solving triangular systems of equations:
 *   - \f$ B \leftarrow \alpha T^{-1} B \f$
 *   - \f$ B \leftarrow \alpha T^{-T} B \f$
 *   - \f$ B \leftarrow \alpha B T^{-1} \f$
 *   - \f$ B \leftarrow \alpha B T^{-T} \f$
 *
 * where \b T is an upper or lower triangular matrix.
 *
 * @param[in] order     Row/column order.
 * @param[in] side      The side of triangular matrix.
 * @param[in] uplo      The triangle in matrix being referenced.
 * @param[in] transA    How matrix \b A is to be transposed.
 * @param[in] diag      Specify whether matrix is unit triangular.
 * @param[in] M         Number of rows in matrices \b A and \b B.
 * @param[in] N         Number of columns in matrices \b A and \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offA      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. For detailed
 *                      description, see clAmdBlasStrsm().
 * @param[out] B        Buffer object storing matrix \b B.
 * @param[in] offB      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. For detailed
 *                      description, see clAmdBlasStrsm().
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *        point arithmetic with double precision;
 *   - \b clAmdBlasInvalidValue if either \b offA or \b offB exceeds the size
 *        of the respective buffer object;
 *   - the same error codes as the clAmdBlasStrsm() function otherwise
 *
 * @ingroup TRSM
 */
clAmdBlasStatus
clAmdBlasZtrsmEx(
    clAmdBlasOrder order,
    clAmdBlasSide side,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    clAmdBlasDiag diag,
    size_t M,
    size_t N,
    DoubleComplex alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    cl_mem B,
    size_t offB,
    size_t ldb,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/*@}*/

/**
 * @defgroup SYRK SYRK - SYmmetric Rank-K update of a matrix
 * @ingroup BLAS3
 */

/*@{*/

/**
 * @brief Rank-k update of a symmetric matrix with float elements.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A A^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T A + \beta C \f$
 *
 * where \b C is a symmetric matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] transA     How matrix \b A is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrix \b A if it is not
 *                       transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrix \b A.
 * @param[in] A          Buffer object storing the matrix \b A.
 * @param[in] lda        Leading dimension of matrix \b A. It cannot be
 *                       less than \b K if \b A matches to the op(\b A) matrix
 *                       in the row-major format, and less than \b N
 *                       otherwise.
 * @param[in] beta       The factor of the matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] ldc        Leading dimension of matric \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasSsyrkEx() instead.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - either \b N or \b K is zero, or
 *     - any of the leading dimensions is invalid;
 *     - the matrix sizes lead to accessing outsize of any of the buffers;
 *   - \b clAmdBlasInvalidMemObject if either \b A or \b C object is
 *     invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs to
 *     was released.
 *
 * @ingroup SYRK
 */
clAmdBlasStatus
clAmdBlasSsyrk(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    size_t N,
    size_t K,
    cl_float alpha,
    const cl_mem A,
    size_t lda,
    cl_float beta,
    cl_mem C,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @example example_ssyrk.c
 * This is an example of how to use the @ref clAmdBlasSsyrk function.
 */

/**
 * @brief Rank-k update of a symmetric matrix with double elements.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A A^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T A + \beta C \f$
 *
 * where \b C is a symmetric matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] transA     How matrix \b A is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrix \b A if it is not
 *                       transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrix \b A.
 * @param[in] A          Buffer object storing the matrix \b A.
 * @param[in] lda        Leading dimension of matrix \b A. For detailed
 *                       description, see clAmdBlasSsyrk().
 * @param[in] beta       The factor of the matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] ldc        Leading dimension of matrix \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasDsyrkEx() instead.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasSsyrk() function otherwise.
 *
 * @ingroup SYRK
 */
clAmdBlasStatus
clAmdBlasDsyrk(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    size_t N,
    size_t K,
    cl_double alpha,
    const cl_mem A,
    size_t lda,
    cl_double beta,
    cl_mem C,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Rank-k update of a symmetric matrix with complex float elements.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A A^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T A + \beta C \f$
 *
 * where \b C is a symmetric matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] transA     How matrix \b A is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrix \b A if it is not
 *                       transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrix \b A.
 * @param[in] A          Buffer object storing the matrix \b A.
 * @param[in] lda        Leading dimension of matrix \b A. For detailed
 *                       description, see clAmdBlasSsyrk().
 * @param[in] beta       The factor of the matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] ldc        Leading dimension of matrix \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasCsyrkEx() instead.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidValue if \b transA is set to \ref clAmdBlasConjTrans.
 *   - the same error codes as the clAmdBlasSsyrk() function otherwise.
 *
 * @ingroup SYRK
 */
clAmdBlasStatus
clAmdBlasCsyrk(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    size_t N,
    size_t K,
    FloatComplex alpha,
    const cl_mem A,
    size_t lda,
    FloatComplex beta,
    cl_mem C,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Rank-k update of a symmetric matrix with complex double elements.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A A^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T A + \beta C \f$
 *
 * where \b C is a symmetric matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] transA     How matrix \b A is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrix \b A if it is not
 *                       transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrix \b A.
 * @param[in] A          Buffer object storing the matrix \b A.
 * @param[in] lda        Leading dimension of matrix \b A. For detailed
 *                       description, see clAmdBlasSsyrk().
 * @param[in] beta       The factor of the matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] ldc        Leading dimension of matrix \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasZsyrkEx() instead.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - \b clAmdBlasInvalidValue if \b transA is set to \ref clAmdBlasConjTrans.
 *   - the same error codes as the clAmdBlasSsyrk() function otherwise.
 *
 * @ingroup SYRK
 */
clAmdBlasStatus
clAmdBlasZsyrk(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    size_t N,
    size_t K,
    DoubleComplex alpha,
    const cl_mem A,
    size_t lda,
    DoubleComplex beta,
    cl_mem C,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Rank-k update of a symmetric matrix with float elements.
 *        Extended version.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A A^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T A + \beta C \f$
 *
 * where \b C is a symmetric matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] transA     How matrix \b A is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrix \b A if it is not
 *                       transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrix \b A.
 * @param[in] A          Buffer object storing the matrix \b A.
 * @param[in] offA       Offset of the first element of the matrix \b A in the
 *                       buffer object. Counted in elements.
 * @param[in] lda        Leading dimension of matrix \b A. It cannot be
 *                       less than \b K if \b A matches to the op(\b A) matrix
 *                       in the row-major format, and less than \b N
 *                       otherwise.
 * @param[in] beta       The factor of the matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] offC       Offset of the first element of the matrix \b C in the
 *                       buffer object. Counted in elements.
 * @param[in] ldc        Leading dimension of matric \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidValue if either \b offA or \b offC exceeds the size
 *        of the respective buffer object;
 *   - the same error codes as the clAmdBlasSsyrk() function otherwise.
 *
 * @ingroup SYRK
 */
clAmdBlasStatus
clAmdBlasSsyrkEx(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    size_t N,
    size_t K,
    cl_float alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    cl_float beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @example example_ssyrk.c
 * This is an example of how to use the @ref clAmdBlasSsyrkEx function.
 */

/**
 * @brief Rank-k update of a symmetric matrix with double elements.
 *        Extended version.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A A^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T A + \beta C \f$
 *
 * where \b C is a symmetric matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] transA     How matrix \b A is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrix \b A if it is not
 *                       transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrix \b A.
 * @param[in] A          Buffer object storing the matrix \b A.
 * @param[in] offA       Offset of the first element of the matrix \b A in the
 *                       buffer object. Counted in elements.
 * @param[in] lda        Leading dimension of matrix \b A. For detailed
 *                       description, see clAmdBlasSsyrk().
 * @param[in] beta       The factor of the matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] offC       Offset of the first element of the matrix \b C in the
 *                       buffer object. Counted in elements.
 * @param[in] ldc        Leading dimension of matrix \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *        point arithmetic with double precision;
 *   - \b clAmdBlasInvalidValue if either \b offA or \b offC exceeds the size
 *        of the respective buffer object;
 *   - the same error codes as the clAmdBlasSsyrk() function otherwise.
 *
 * @ingroup SYRK
 */
clAmdBlasStatus
clAmdBlasDsyrkEx(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    size_t N,
    size_t K,
    cl_double alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    cl_double beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Rank-k update of a symmetric matrix with complex float elements.
 *        Extended version.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A A^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T A + \beta C \f$
 *
 * where \b C is a symmetric matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] transA     How matrix \b A is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrix \b A if it is not
 *                       transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrix \b A.
 * @param[in] A          Buffer object storing the matrix \b A.
 * @param[in] offA       Offset of the first element of the matrix \b A in the
 *                       buffer object. Counted in elements.
 * @param[in] lda        Leading dimension of matrix \b A. For detailed
 *                       description, see clAmdBlasSsyrk().
 * @param[in] beta       The factor of the matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] offC       Offset of the first element of the matrix \b C in the
 *                       buffer object. Counted in elements.
 * @param[in] ldc        Leading dimension of matrix \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidValue if either \b offA or \b offC exceeds the size
 *        of the respective buffer object;
 *   - \b clAmdBlasInvalidValue if \b transA is set to \ref clAmdBlasConjTrans.
 *   - the same error codes as the clAmdBlasSsyrk() function otherwise.
 *
 * @ingroup SYRK
 */
clAmdBlasStatus
clAmdBlasCsyrkEx(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    size_t N,
    size_t K,
    FloatComplex alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    FloatComplex beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Rank-k update of a symmetric matrix with complex double elements.
 *        Extended version.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A A^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T A + \beta C \f$
 *
 * where \b C is a symmetric matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] transA     How matrix \b A is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrix \b A if it is not
 *                       transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrix \b A.
 * @param[in] A          Buffer object storing the matrix \b A.
 * @param[in] offA       Offset of the first element of the matrix \b A in the
 *                       buffer object. Counted in elements.
 * @param[in] lda        Leading dimension of matrix \b A. For detailed
 *                       description, see clAmdBlasSsyrk().
 * @param[in] beta       The factor of the matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] offC       Offset of the first element of the matrix \b C in the
 *                       buffer object. Counted in elements.
 * @param[in] ldc        Leading dimension of matrix \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *         point arithmetic with double precision;
 *   - \b clAmdBlasInvalidValue if either \b offA or \b offC exceeds the size
 *        of the respective buffer object;
 *   - \b clAmdBlasInvalidValue if \b transA is set to \ref clAmdBlasConjTrans.
 *   - the same error codes as the clAmdBlasSsyrk() function otherwise.
 *
 * @ingroup SYRK
 */
clAmdBlasStatus
clAmdBlasZsyrkEx(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    size_t N,
    size_t K,
    DoubleComplex alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    DoubleComplex beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/*@}*/

/**
 * @defgroup SYR2K SYR2K - SYmmetric Rank-2K update to a matrix
 * @ingroup BLAS3
 */

/*@{*/

/**
 * @brief Rank-2k update of a symmetric matrix with float elements.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A B^T + \alpha B A^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B + \alpha B^T A \beta C \f$
 *
 * where \b C is a symmetric matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] transAB    How matrices \b A and \b B is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrices \b A and \b B if they
 *                       are not transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrices \b A and \b B.
 * @param[in] A          Buffer object storing matrix \b A.
 * @param[in] lda        Leading dimension of matrix \b A. It cannot be less
 *                       than \b K if \b A matches to the op(\b A) matrix
 *                       in the row-major format, and less than \b N
 *                       otherwise.
 * @param[in] B          Buffer object storing matrix \b B.
 * @param[in] ldb        Leading dimension of matrix \b B. It cannot be less
 *                       less than \b K if \b B matches to the op(\b B) matrix
 *                       in the row-major format, and less than \b N
 *                       otherwise.
 * @param[in] beta       The factor of matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] ldc        Leading dimension of matrix \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasSsyr2kEx() instead.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - either \b N or \b K is zero, or
 *     - any of the leading dimensions is invalid;
 *     - the matrix sizes lead to accessing outsize of any of the buffers;
 *   - \b clAmdBlasInvalidMemObject if either \b A, \b B or \b C object is
 *     invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs to
 *     was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup SYR2K
 */
clAmdBlasStatus
clAmdBlasSsyr2k(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transAB,
    size_t N,
    size_t K,
    cl_float alpha,
    const cl_mem A,
    size_t lda,
    const cl_mem B,
    size_t ldb,
    cl_float beta,
    cl_mem C,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @example example_ssyr2k.c
 * This is an example of how to use the @ref clAmdBlasSsyr2k function.
 */

/**
 * @brief Rank-2k update of a symmetric matrix with double elements.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A B^T + \alpha B A^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B + \alpha B^T A \beta C \f$
 *
 * where \b C is a symmetric matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] transAB    How matrices \b A and \b B is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrices \b A and \b B if they
 *                       are not transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrices \b A and \b B.
 * @param[in] A          Buffer object storing matrix \b A.
 * @param[in] lda        Leading dimension of matrix \b A. For detailed
 *                       description, see clAmdBlasSsyr2k().
 * @param[in] B          Buffer object storing matrix \b B.
 * @param[in] ldb        Leading dimension of matrix \b B. For detailed
 *                       description, see clAmdBlasSsyr2k().
 * @param[in] beta       The factor of matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] ldc        Leading dimension of matrix \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasDsyr2kEx() instead.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasSsyr2k() function otherwise.
 *
 * @ingroup SYR2K
 */
clAmdBlasStatus
clAmdBlasDsyr2k(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transAB,
    size_t N,
    size_t K,
    cl_double alpha,
    const cl_mem A,
    size_t lda,
    const cl_mem B,
    size_t ldb,
    cl_double beta,
    cl_mem C,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Rank-2k update of a symmetric matrix with complex float elements.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A B^T + \alpha B A^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B + \alpha B^T A \beta C \f$
 *
 * where \b C is a symmetric matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] transAB    How matrices \b A and \b B is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrices \b A and \b B if they
 *                       are not transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrices \b A and \b B.
 * @param[in] A          Buffer object storing matrix \b A.
 * @param[in] lda        Leading dimension of matrix \b A. For detailed
 *                       description, see clAmdBlasSsyr2k().
 * @param[in] B          Buffer object storing matrix \b B.
 * @param[in] ldb        Leading dimension of matrix \b B. For detailed
 *                       description, see clAmdBlasSsyr2k().
 * @param[in] beta       The factor of matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] ldc        Leading dimension of matrix \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasCsyr2kEx() instead.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidValue if \b transAB is set to \ref clAmdBlasConjTrans.
 *   - the same error codes as the clAmdBlasSsyr2k() function otherwise.
 *
 * @ingroup SYR2K
 */
clAmdBlasStatus
clAmdBlasCsyr2k(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transAB,
    size_t N,
    size_t K,
    FloatComplex alpha,
    const cl_mem A,
    size_t lda,
    const cl_mem B,
    size_t ldb,
    FloatComplex beta,
    cl_mem C,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Rank-2k update of a symmetric matrix with complex double elements.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A B^T + \alpha B A^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B + \alpha B^T A \beta C \f$
 *
 * where \b C is a symmetric matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] transAB    How matrices \b A and \b B is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrices \b A and \b B if they
 *                       are not transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrices \b A and \b B.
 * @param[in] A          Buffer object storing matrix \b A.
 * @param[in] lda        Leading dimension of matrix \b A. For detailed
 *                       description, see clAmdBlasSsyr2k().
 * @param[in] B          Buffer object storing matrix \b B.
 * @param[in] ldb        Leading dimension of matrix \b B. For detailed
 *                       description, see clAmdBlasSsyr2k().
 * @param[in] beta       The factor of matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] ldc        Leading dimension of matrix \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * The function is obsolete and is not recommended for using in new
 * applications. Use the superseding function clAmdBlasZsyr2kEx() instead.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - \b clAmdBlasInvalidValue if \b transAB is set to \ref clAmdBlasConjTrans.
 *   - the same error codes as the clAmdBlasSsyr2k() function otherwise.
 *
 * @ingroup SYR2K
 */
clAmdBlasStatus
clAmdBlasZsyr2k(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transAB,
    size_t N,
    size_t K,
    DoubleComplex alpha,
    const cl_mem A,
    size_t lda,
    const cl_mem B,
    size_t ldb,
    DoubleComplex beta,
    cl_mem C,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Rank-2k update of a symmetric matrix with float elements.
 *        Extended version.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A B^T + \alpha B A^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B + \alpha B^T A \beta C \f$
 *
 * where \b C is a symmetric matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] transAB    How matrices \b A and \b B is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrices \b A and \b B if they
 *                       are not transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrices \b A and \b B.
 * @param[in] A          Buffer object storing matrix \b A.
 * @param[in] offA       Offset of the first element of the matrix \b A in the
 *                       buffer object. Counted in elements.
 * @param[in] lda        Leading dimension of matrix \b A. It cannot be less
 *                       than \b K if \b A matches to the op(\b A) matrix
 *                       in the row-major format, and less than \b N
 *                       otherwise.
 * @param[in] B          Buffer object storing matrix \b B.
 * @param[in] offB       Offset of the first element of the matrix \b B in the
 *                       buffer object. Counted in elements.
 * @param[in] ldb        Leading dimension of matrix \b B. It cannot be less
 *                       less than \b K if \b B matches to the op(\b B) matrix
 *                       in the row-major format, and less than \b N
 *                       otherwise.
 * @param[in] beta       The factor of matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] offC       Offset of the first element of the matrix \b C in the
 *                       buffer object. Counted in elements.
 * @param[in] ldc        Leading dimension of matrix \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidValue if either \b offA, \b offB or \b offC exceeds
 *        the size of the respective buffer object;
 *   - the same error codes as the clAmdBlasSsyr2k() function otherwise.
 *
 * @ingroup SYR2K
 */
clAmdBlasStatus
clAmdBlasSsyr2kEx(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transAB,
    size_t N,
    size_t K,
    cl_float alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem B,
    size_t offB,
    size_t ldb,
    cl_float beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @example example_ssyr2k.c
 * This is an example of how to use the @ref clAmdBlasSsyr2kEx function.
 */

/**
 * @brief Rank-2k update of a symmetric matrix with double elements.
 *        Extended version.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A B^T + \alpha B A^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B + \alpha B^T A \beta C \f$
 *
 * where \b C is a symmetric matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] transAB    How matrices \b A and \b B is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrices \b A and \b B if they
 *                       are not transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrices \b A and \b B.
 * @param[in] A          Buffer object storing matrix \b A.
 * @param[in] offA       Offset of the first element of the matrix \b A in the
 *                       buffer object. Counted in elements.
 * @param[in] lda        Leading dimension of matrix \b A. For detailed
 *                       description, see clAmdBlasSsyr2k().
 * @param[in] B          Buffer object storing matrix \b B.
 * @param[in] offB       Offset of the first element of the matrix \b B in the
 *                       buffer object. Counted in elements.
 * @param[in] ldb        Leading dimension of matrix \b B. For detailed
 *                       description, see clAmdBlasSsyr2k().
 * @param[in] beta       The factor of matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] offC       Offset of the first element of the matrix \b C in the
 *                       buffer object. Counted in elements.
 * @param[in] ldc        Leading dimension of matrix \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *        point arithmetic with double precision;
 *   - \b clAmdBlasInvalidValue if either \b offA, \b offB or \b offC exceeds
 *        the size of the respective buffer object;
 *   - the same error codes as the clAmdBlasSsyr2k() function otherwise.
 *
 * @ingroup SYR2K
 */
clAmdBlasStatus
clAmdBlasDsyr2kEx(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transAB,
    size_t N,
    size_t K,
    cl_double alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem B,
    size_t offB,
    size_t ldb,
    cl_double beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Rank-2k update of a symmetric matrix with complex float elements.
 *        Extended version.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A B^T + \alpha B A^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B + \alpha B^T A \beta C \f$
 *
 * where \b C is a symmetric matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] transAB    How matrices \b A and \b B is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrices \b A and \b B if they
 *                       are not transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrices \b A and \b B.
 * @param[in] A          Buffer object storing matrix \b A.
 * @param[in] offA       Offset of the first element of the matrix \b A in the
 *                       buffer object. Counted in elements.
 * @param[in] lda        Leading dimension of matrix \b A. For detailed
 *                       description, see clAmdBlasSsyr2k().
 * @param[in] B          Buffer object storing matrix \b B.
 * @param[in] offB       Offset of the first element of the matrix \b B in the
 *                       buffer object. Counted in elements.
 * @param[in] ldb        Leading dimension of matrix \b B. For detailed
 *                       description, see clAmdBlasSsyr2k().
 * @param[in] beta       The factor of matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] offC       Offset of the first element of the matrix \b C in the
 *                       buffer object. Counted in elements.
 * @param[in] ldc        Leading dimension of matrix \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidValue if either \b offA, \b offB or \b offC exceeds
 *        the size of the respective buffer object;
 *   - \b clAmdBlasInvalidValue if \b transAB is set to \ref clAmdBlasConjTrans.
 *   - the same error codes as the clAmdBlasSsyr2k() function otherwise.
 *
 * @ingroup SYR2K
 */
clAmdBlasStatus
clAmdBlasCsyr2kEx(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transAB,
    size_t N,
    size_t K,
    FloatComplex alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem B,
    size_t offB,
    size_t ldb,
    FloatComplex beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Rank-2k update of a symmetric matrix with complex double elements.
 *        Extended version.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A B^T + \alpha B A^T + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^T B + \alpha B^T A \beta C \f$
 *
 * where \b C is a symmetric matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] transAB    How matrices \b A and \b B is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrices \b A and \b B if they
 *                       are not transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrices \b A and \b B.
 * @param[in] A          Buffer object storing matrix \b A.
 * @param[in] offA       Offset of the first element of the matrix \b A in the
 *                       buffer object. Counted in elements.
 * @param[in] lda        Leading dimension of matrix \b A. For detailed
 *                       description, see clAmdBlasSsyr2k().
 * @param[in] B          Buffer object storing matrix \b B.
 * @param[in] offB       Offset of the first element of the matrix \b B in the
 *                       buffer object. Counted in elements.
 * @param[in] ldb        Leading dimension of matrix \b B. For detailed
 *                       description, see clAmdBlasSsyr2k().
 * @param[in] beta       The factor of matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] offC       Offset of the first element of the matrix \b C in the
 *                       buffer object. Counted in elements.
 * @param[in] ldc        Leading dimension of matrix \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *        point arithmetic with double precision;
 *   - \b clAmdBlasInvalidValue if either \b offA, \b offB or \b offC exceeds
 *        the size of the respective buffer object;
 *   - \b clAmdBlasInvalidValue if \b transAB is set to \ref clAmdBlasConjTrans.
 *   - the same error codes as the clAmdBlasSsyr2k() function otherwise.
 *
 * @ingroup SYR2K
 */
clAmdBlasStatus
clAmdBlasZsyr2kEx(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transAB,
    size_t N,
    size_t K,
    DoubleComplex alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem B,
    size_t offB,
    size_t ldb,
    DoubleComplex beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/*@}*/


/**
 * @defgroup SYMM SYMM  - SYmmetric Matrix-matrix Multiply
 * @ingroup BLAS3
 */
/*@{*/

/**
 * @brief Matrix-matrix product of symmetric rectangular matrices with float
 * elements.
 *
 * Matrix-matrix products:
 *   - \f$ C \leftarrow \alpha A B + \beta C \f$
 *   - \f$ C \leftarrow \alpha B A + \beta C \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] side		The side of triangular matrix.
 * @param[in] uplo		The triangle in matrix being referenced.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offa      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b M when the \b side parameter is set to
 *                      \b clAmdBlasLeft,\n or less than \b N when the
 *                      parameter is set to \b clAmdBlasRight.
 * @param[in] B         Buffer object storing matrix \b B.
 * @param[in] offb      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M
 *                      when it is set to \b clAmdBlasColumnMajor.
 * @param[in] beta      The factor of matrix \b C.
 * @param[out] C        Buffer object storing matrix \b C.
 * @param[in] offc      Offset of the first element of the matrix \b C in the
 *                      buffer object. Counted in elements.
 * @param[in] ldc       Leading dimension of matrix \b C. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M when
 *                      it is set to \b clAmdBlasColumnMajorOrder.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events			  Event objects per each command queue that identify
 *								  a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - \b M or \b N is zero, or
 *     - any of the leading dimensions is invalid;
 *     - the matrix sizes lead to accessing outsize of any of the buffers;
 *   - \b clAmdBlasInvalidMemObject if A, B, or C object is invalid,
 *     or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfResources if you use image-based function implementation
 *     and no suitable scratch image available;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs to
 *     was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup SYMM
 */
clAmdBlasStatus
clAmdBlasSsymm(
    clAmdBlasOrder order,
    clAmdBlasSide side,
    clAmdBlasUplo uplo,
    size_t M,
    size_t N,
    cl_float alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem B,
    size_t offb,
    size_t ldb,
    cl_float beta,
    cl_mem C,
    size_t offc,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/**
 * @example example_ssymm.c
 * This is an example of how to use the @ref clAmdBlasSsymm function.
 */


/**
 * @brief Matrix-matrix product of symmetric rectangular matrices with double
 * elements.
 *
 * Matrix-matrix products:
 *   - \f$ C \leftarrow \alpha A B + \beta C \f$
 *   - \f$ C \leftarrow \alpha B A + \beta C \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] side		The side of triangular matrix.
 * @param[in] uplo		The triangle in matrix being referenced.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offa      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b M when the \b side parameter is set to
 *                      \b clAmdBlasLeft,\n or less than \b N when the
 *                      parameter is set to \b clAmdBlasRight.
 * @param[in] B         Buffer object storing matrix \b B.
 * @param[in] offb      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M
 *                      when it is set to \b clAmdBlasColumnMajor.
 * @param[in] beta      The factor of matrix \b C.
 * @param[out] C        Buffer object storing matrix \b C.
 * @param[in] offc      Offset of the first element of the matrix \b C in the
 *                      buffer object. Counted in elements.
 * @param[in] ldc       Leading dimension of matrix \b C. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M when
 *                      it is set to \b clAmdBlasColumnMajorOrder.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events			  Event objects per each command queue that identify
 *								  a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasSsymm() function otherwise.
 *
 * @ingroup SYMM
 */
clAmdBlasStatus
clAmdBlasDsymm(
    clAmdBlasOrder order,
    clAmdBlasSide side,
    clAmdBlasUplo uplo,
    size_t M,
    size_t N,
    cl_double alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem B,
    size_t offb,
    size_t ldb,
    cl_double beta,
    cl_mem C,
    size_t offc,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);


/**
 * @brief Matrix-matrix product of symmetric rectangular matrices with
 * float-complex elements.
 *
 * Matrix-matrix products:
 *   - \f$ C \leftarrow \alpha A B + \beta C \f$
 *   - \f$ C \leftarrow \alpha B A + \beta C \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] side		The side of triangular matrix.
 * @param[in] uplo		The triangle in matrix being referenced.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offa      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b M when the \b side parameter is set to
 *                      \b clAmdBlasLeft,\n or less than \b N when the
 *                      parameter is set to \b clAmdBlasRight.
 * @param[in] B         Buffer object storing matrix \b B.
 * @param[in] offb      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M
 *                      when it is set to \b clAmdBlasColumnMajor.
 * @param[in] beta      The factor of matrix \b C.
 * @param[out] C        Buffer object storing matrix \b C.
 * @param[in] offc      Offset of the first element of the matrix \b C in the
 *                      buffer object. Counted in elements.
 * @param[in] ldc       Leading dimension of matrix \b C. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M when
 *                      it is set to \b clAmdBlasColumnMajorOrder.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events			  Event objects per each command queue that identify
 *								  a particular kernel execution instance.
 *
 * @return The same result as the clAmdBlasSsymm() function.
 *
 * @ingroup SYMM
 */
clAmdBlasStatus
clAmdBlasCsymm(
    clAmdBlasOrder order,
    clAmdBlasSide side,
    clAmdBlasUplo uplo,
    size_t M,
    size_t N,
    cl_float2 alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem B,
    size_t offb,
    size_t ldb,
    cl_float2 beta,
    cl_mem C,
    size_t offc,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);

/**
 * @brief Matrix-matrix product of symmetric rectangular matrices with
 * double-complex elements.
 *
 * Matrix-matrix products:
 *   - \f$ C \leftarrow \alpha A B + \beta C \f$
 *   - \f$ C \leftarrow \alpha B A + \beta C \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] side		The side of triangular matrix.
 * @param[in] uplo		The triangle in matrix being referenced.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offa      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b M when the \b side parameter is set to
 *                      \b clAmdBlasLeft,\n or less than \b N when the
 *                      parameter is set to \b clAmdBlasRight.
 * @param[in] B         Buffer object storing matrix \b B.
 * @param[in] offb      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M
 *                      when it is set to \b clAmdBlasColumnMajor.
 * @param[in] beta      The factor of matrix \b C.
 * @param[out] C        Buffer object storing matrix \b C.
 * @param[in] offc      Offset of the first element of the matrix \b C in the
 *                      buffer object. Counted in elements.
 * @param[in] ldc       Leading dimension of matrix \b C. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M when
 *                      it is set to \b clAmdBlasColumnMajorOrder.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events			  Event objects per each command queue that identify
 *								  a particular kernel execution instance.
 *
 * @return The same result as the clAmdBlasDsymm() function.
 *
 * @ingroup SYMM
 */
clAmdBlasStatus
clAmdBlasZsymm(
    clAmdBlasOrder order,
    clAmdBlasSide side,
    clAmdBlasUplo uplo,
    size_t M,
    size_t N,
    cl_double2 alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem B,
    size_t offb,
    size_t ldb,
    cl_double2 beta,
    cl_mem C,
    size_t offc,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/*@}*/


/**
 * @defgroup HEMM HEMM  - HErmitian Matrix-matrix Multiplication
 * @ingroup BLAS3
 */
/*@{*/

/**
 * @brief Matrix-matrix product of hermitian rectangular matrices with
 * float-complex elements.
 *
 * Matrix-matrix products:
 *   - \f$ C \leftarrow \alpha A B + \beta C \f$
 *   - \f$ C \leftarrow \alpha B A + \beta C \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] side		The side of triangular matrix.
 * @param[in] uplo		The triangle in matrix being referenced.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offa      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b M when the \b side parameter is set to
 *                      \b clAmdBlasLeft,\n or less than \b N when the
 *                      parameter is set to \b clAmdBlasRight.
 * @param[in] B         Buffer object storing matrix \b B.
 * @param[in] offb      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M
 *                      when it is set to \b clAmdBlasColumnMajor.
 * @param[in] beta      The factor of matrix \b C.
 * @param[out] C        Buffer object storing matrix \b C.
 * @param[in] offc      Offset of the first element of the matrix \b C in the
 *                      buffer object. Counted in elements.
 * @param[in] ldc       Leading dimension of matrix \b C. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M when
 *                      it is set to \b clAmdBlasColumnMajorOrder.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - \b M or \b N is zero, or
 *     - any of the leading dimensions is invalid;
 *     - the matrix sizes lead to accessing outsize of any of the buffers;
 *   - \b clAmdBlasInvalidMemObject if A, B, or C object is invalid,
 *     or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfResources if you use image-based function implementation
 *     and no suitable scratch image available;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs to
 *     was released;
 *   - \b clAmdBlasInvalidOperation if kernel compilation relating to a previous
 *     call has not completed for any of the target devices;
 *   - \b clAmdBlasCompilerNotAvailable if a compiler is not available;
 *   - \b clAmdBlasBuildProgramFailure if there is a failure to build a program
 *     executable.
 *
 * @ingroup HEMM
 */
clAmdBlasStatus
clAmdBlasChemm(
    clAmdBlasOrder order,
    clAmdBlasSide side,
    clAmdBlasUplo uplo,
    size_t M,
    size_t N,
    cl_float2 alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem B,
    size_t offb,
    size_t ldb,
    cl_float2 beta,
    cl_mem C,
    size_t offc,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/**
 * @example example_chemm.cpp
 * This is an example of how to use the @ref clAmdBlasChemm function.
 */


/**
 * @brief Matrix-matrix product of hermitian rectangular matrices with
 * double-complex elements.
 *
 * Matrix-matrix products:
 *   - \f$ C \leftarrow \alpha A B + \beta C \f$
 *   - \f$ C \leftarrow \alpha B A + \beta C \f$
 *
 * @param[in] order     Row/column order.
 * @param[in] side		The side of triangular matrix.
 * @param[in] uplo		The triangle in matrix being referenced.
 * @param[in] M         Number of rows in matrix \b A.
 * @param[in] N         Number of columns in matrix \b B.
 * @param[in] alpha     The factor of matrix \b A.
 * @param[in] A         Buffer object storing matrix \b A.
 * @param[in] offa      Offset of the first element of the matrix \b A in the
 *                      buffer object. Counted in elements.
 * @param[in] lda       Leading dimension of matrix \b A. It cannot be less
 *                      than \b M when the \b side parameter is set to
 *                      \b clAmdBlasLeft,\n or less than \b N when the
 *                      parameter is set to \b clAmdBlasRight.
 * @param[in] B         Buffer object storing matrix \b B.
 * @param[in] offb      Offset of the first element of the matrix \b B in the
 *                      buffer object. Counted in elements.
 * @param[in] ldb       Leading dimension of matrix \b B. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M
 *                      when it is set to \b clAmdBlasColumnMajor.
 * @param[in] beta      The factor of matrix \b C.
 * @param[out] C        Buffer object storing matrix \b C.
 * @param[in] offc      Offset of the first element of the matrix \b C in the
 *                      buffer object. Counted in elements.
 * @param[in] ldc       Leading dimension of matrix \b C. It cannot be less
 *                      than \b N when the \b order parameter is set to
 *                      \b clAmdBlasRowMajor,\n or less than \b M when
 *                      it is set to \b clAmdBlasColumnMajorOrder.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasChemm() function otherwise.
 *
 * @ingroup HEMM
 */
clAmdBlasStatus
clAmdBlasZhemm(
    clAmdBlasOrder order,
    clAmdBlasSide side,
    clAmdBlasUplo uplo,
    size_t M,
    size_t N,
    cl_double2 alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem B,
    size_t offb,
    size_t ldb,
    cl_double2 beta,
    cl_mem C,
    size_t offc,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/*@}*/


/**
 * @defgroup HERK HERK  - HErmitian Rank-K update to a matrix
 * @ingroup BLAS3
 */
/*@{*/

/**
 * @brief Rank-k update of a hermitian matrix with float-complex elements.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A A^H + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^H A + \beta C \f$
 *
 * where \b C is a hermitian matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] transA     How matrix \b A is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrix \b A if it is not
 *                       transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrix \b A.
 * @param[in] A          Buffer object storing the matrix \b A.
 * @param[in] offa       Offset in number of elements for the first element in matrix \b A.
 * @param[in] lda        Leading dimension of matrix \b A. It cannot be
 *                       less than \b K if \b A matches to the op(\b A) matrix
 *                       in the row-major format, and less than \b N
 *                       otherwise.
 * @param[in] beta       The factor of the matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] offc       Offset in number of elements for the first element in matrix \b C.
 * @param[in] ldc        Leading dimension of matric \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - either \b N or \b K is zero, or
 *     - any of the leading dimensions is invalid;
 *     - the matrix sizes lead to accessing outsize of any of the buffers;
 *   - \b clAmdBlasInvalidMemObject if either \b A or \b C object is
 *     invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs to
 *     was released.
 *
 * @ingroup HERK
 */
clAmdBlasStatus
clAmdBlasCherk(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    size_t N,
    size_t K,
    float alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    float beta,
    cl_mem C,
    size_t offc,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/**
 * @example example_cherk.cpp
 * This is an example of how to use the @ref clAmdBlasCherk function.
 */


/**
 * @brief Rank-k update of a hermitian matrix with double-complex elements.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A A^H + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^H A + \beta C \f$
 *
 * where \b C is a hermitian matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] transA     How matrix \b A is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrix \b A if it is not
 *                       transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrix \b A.
 * @param[in] A          Buffer object storing the matrix \b A.
 * @param[in] offa       Offset in number of elements for the first element in matrix \b A.
 * @param[in] lda        Leading dimension of matrix \b A. It cannot be
 *                       less than \b K if \b A matches to the op(\b A) matrix
 *                       in the row-major format, and less than \b N
 *                       otherwise.
 * @param[in] beta       The factor of the matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] offc       Offset in number of elements for the first element in matrix \b C.
 * @param[in] ldc        Leading dimension of matric \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasCherk() function otherwise.
 *
 * @ingroup HERK
 */
clAmdBlasStatus
clAmdBlasZherk(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose transA,
    size_t N,
    size_t K,
    double alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    double beta,
    cl_mem C,
    size_t offc,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/*@}*/


/**
 * @defgroup HER2K HER2K  - HErmitian rank-2K update to a matrix
 * @ingroup BLAS3
 */
/*@{*/

/**
 * @brief Rank-2k update of a hermitian matrix with float-complex elements.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A B^H + conj( \alpha ) B A^H + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^H B + conj( \alpha ) B^H A + \beta C \f$
 *
 * where \b C is a hermitian matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] trans      How matrix \b A is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrix \b A if it is not
 *                       transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrix \b A.
 * @param[in] A          Buffer object storing the matrix \b A.
 * @param[in] offa       Offset in number of elements for the first element in matrix \b A.
 * @param[in] lda        Leading dimension of matrix \b A. It cannot be
 *                       less than \b K if \b A matches to the op(\b A) matrix
 *                       in the row-major format, and less than \b N
 *                       otherwise. Vive-versa for transpose case
 * @param[in] B          Buffer object storing the matrix \b B.
 * @param[in] offb       Offset in number of elements for the first element in matrix \b B.
 * @param[in] ldb        Leading dimension of matrix \b B. It cannot be
 *                       less than \b K if \b A matches to the op(\b A) matrix
 *                       in the row-major format, and less than \b N
 *                       otherwise. Vive-versa for transpose case
 * @param[in] beta       The factor of the matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] offc       Offset in number of elements for the first element in matrix \b C.
 * @param[in] ldc        Leading dimension of matric \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasNotInitialized if clAmdBlasSetup() was not called;
 *   - \b clAmdBlasInvalidValue if invalid parameters are passed:
 *     - either \b N or \b K is zero, or
 *     - any of the leading dimensions is invalid;
 *     - the matrix sizes lead to accessing outsize of any of the buffers;
 *   - \b clAmdBlasInvalidMemObject if either \b A , \b B or \b C object is
 *     invalid, or an image object rather than the buffer one;
 *   - \b clAmdBlasOutOfHostMemory if the library can't allocate memory for
 *     internal structures;
 *   - \b clAmdBlasInvalidCommandQueue if the passed command queue is invalid;
 *   - \b clAmdBlasInvalidContext if a context a passed command queue belongs to
 *     was released.
 *
 * @ingroup HER2K
 */
clAmdBlasStatus
clAmdBlasCher2k(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose trans,
    size_t N,
    size_t K,
    FloatComplex alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem B,
    size_t offb,
    size_t ldb,
    cl_float beta,
    cl_mem C,
    size_t offc,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/**
 * @example example_cher2k.c
 * This is an example of how to use the @ref clAmdBlasCher2k function.
 */


/**
 * @brief Rank-2k update of a hermitian matrix with double-complex elements.
 *
 * Rank-k updates:
 *   - \f$ C \leftarrow \alpha A B^H + conj( \alpha ) B A^H + \beta C \f$
 *   - \f$ C \leftarrow \alpha A^H B + conj( \alpha ) B^H A + \beta C \f$
 *
 * where \b C is a hermitian matrix.
 *
 * @param[in] order      Row/column order.
 * @param[in] uplo       The triangle in matrix \b C being referenced.
 * @param[in] trans      How matrix \b A is to be transposed.
 * @param[in] N          Number of rows and columns in matrix \b C.
 * @param[in] K          Number of columns of the matrix \b A if it is not
 *                       transposed, and number of rows otherwise.
 * @param[in] alpha      The factor of matrix \b A.
 * @param[in] A          Buffer object storing the matrix \b A.
 * @param[in] offa       Offset in number of elements for the first element in matrix \b A.
 * @param[in] lda        Leading dimension of matrix \b A. It cannot be
 *                       less than \b K if \b A matches to the op(\b A) matrix
 *                       in the row-major format, and less than \b N
 *                       otherwise. Vive-versa for transpose case
 * @param[in] B          Buffer object storing the matrix \b B.
 * @param[in] offb       Offset in number of elements for the first element in matrix \b B.
 * @param[in] ldb        Leading dimension of matrix \b B. It cannot be
 *                       less than \b K if \b A matches to the op(\b A) matrix
 *                       in the row-major format, and less than \b N
 *                       otherwise. Vive-versa for transpose case
 * @param[in] beta       The factor of the matrix \b C.
 * @param[out] C         Buffer object storing matrix \b C.
 * @param[in] offc       Offset in number of elements for the first element in matrix \b C.
 * @param[in] ldc        Leading dimension of matric \b C. It cannot be less
 *                       than \b N.
 * @param[in] numCommandQueues    Number of OpenCL command queues in which the
 *                                task is to be performed.
 * @param[in] commandQueues       OpenCL command queues.
 * @param[in] numEventsInWaitList Number of events in the event wait list.
 * @param[in] eventWaitList       Event wait list.
 * @param[in] events     Event objects per each command queue that identify
 *                       a particular kernel execution instance.
 *
 * @return
 *   - \b clAmdBlasSuccess on success;
 *   - \b clAmdBlasInvalidDevice if a target device does not support floating
 *     point arithmetic with double precision;
 *   - the same error codes as the clAmdBlasCher2k() function otherwise.
 *
 * @ingroup HER2K
 */
clAmdBlasStatus
clAmdBlasZher2k(
    clAmdBlasOrder order,
    clAmdBlasUplo uplo,
    clAmdBlasTranspose trans,
    size_t N,
    size_t K,
    DoubleComplex alpha,
    const cl_mem A,
    size_t offa,
    size_t lda,
    const cl_mem B,
    size_t offb,
    size_t ldb,
    cl_double beta,
    cl_mem C,
    size_t offc,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events);
/*@}*/




#ifdef __cplusplus
}      /* extern "C" { */
#endif

#endif /* CLAMDBLAS_H_ */
