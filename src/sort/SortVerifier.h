#ifndef SORTVERIFIER_H
#define SORTVERIFIER_H

template <typename T, size_t count, template <typename, size_t> class A>
class SortVerifier
{
    public:
        static bool verify(A<T, count>* alg, T* data, T* result)
        {
            if(alg->isInPlace())
                result = data;

            for(size_t i = 0; i < count - 1; i++)
                if(result[i] > result[i + 1])
                    return false;

            return true;
        }
};

#endif // SORTVERIFIER_H
