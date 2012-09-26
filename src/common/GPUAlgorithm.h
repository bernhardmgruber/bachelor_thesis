#ifndef GPUALGORITHM_H
#define GPUALGORITHM_H

class GPUAlgorithm
{
    public:
        virtual const string getName() = 0;
        virtual void init(Context* context) = 0;
        virtual void upload(Context* context, CommandQueue* queue, size_t workGroupSize, void* data, size_t size) = 0;
        virtual void run(CommandQueue* queue, size_t workGroupSize, size_t size) = 0;
        virtual void download(CommandQueue* queue, void* result, size_t size) = 0;
        virtual void cleanup() = 0;
};

#endif // GPUALGORITHM_H
