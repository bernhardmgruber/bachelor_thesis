#ifndef MATRIXVERIFIER_H
#define MATRIXVERIFIER_H

template <typename T, size_t count, template <typename, size_t> class A>
class MatrixVerifier
{
    public:
        static bool verify(A<T, count>* alg, T* data, T* result)
        {
            return false;
        }
};

#endif // MATRIXVERIFIER_H
