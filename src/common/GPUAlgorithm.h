#ifndef GPUALGORITHM_H
#define GPUALGORITHM_H

template<typename T, size_t count>
class GPUAlgorithm
{
    public:
        virtual string getName() = 0;
        virtual void init(Context* context) = 0;
        virtual void upload(Context* context, size_t workGroupSize, T* data) = 0;
        virtual void run(CommandQueue* queue, size_t workGroupSize) = 0;
        virtual void download(CommandQueue* queue, T* result) = 0;
        virtual void cleanup() = 0;
};

#endif // GPUALGORITHM_H
