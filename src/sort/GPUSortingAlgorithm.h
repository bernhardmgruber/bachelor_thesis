#ifndef GPUSORTINGALGORITHM_H
#define GPUSORTINGALGORITHM_H

template<typename T, size_t count>
class GPUSortingAlgorithm
{
    public:
        virtual string getName() = 0;
        virtual void init(Context* context) = 0;
        virtual void upload(Context* context, T* data) = 0;
        virtual void sort(CommandQueue* queue, size_t workGroupSize) = 0;
        virtual void download(CommandQueue* queue, T* data) = 0;
        virtual void cleanup() = 0;
};

#endif // GPUSORTINGALGORITHM_H
