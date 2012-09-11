#ifndef GPUSCANALGORITHM_H
#define GPUSCANALGORITHM_H

template<typename T, size_t count>
class GPUScanAlgorithm
{
    public:
        virtual string getName() = 0;
        virtual bool isInclusiv() = 0;
        virtual void init(Context* context) = 0;
        virtual void upload(Context* context, size_t workGroupSize, T* data) = 0;
        virtual void scan(CommandQueue* queue, size_t workGroupSize) = 0;
        virtual void download(CommandQueue* queue, T* result) = 0;
        virtual void cleanup() = 0;
};


#endif // GPUSCANALGORITHM_H
