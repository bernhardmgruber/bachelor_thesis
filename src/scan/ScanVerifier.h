#ifndef SCANVERIFIER_H
#define SCANVERIFIER_H

template <typename T, size_t count, template <typename, size_t> class A>
class ScanVerifier
{
    public:
        static bool verify(A<T, count>* alg, T* data, T* result)
        {
            if(alg->isInclusiv())
            {
                if(data[0] != result[0])
                    return false;

                for(size_t i = 1; i < count; i++)
                    if(result[i] !=  result[i - 1] + data[i])
                        return false;

                return true;
            }
            else
            {
                if(result[0] != 0)
                    return false;

                for(size_t i = 1; i < count; i++)
                    if(result[i] != result[i - 1] + data[i - 1])
                        return false;

                return true;
            }
        }
};

#endif // SCANVERIFIER_H
