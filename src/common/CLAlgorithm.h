#pragma once

#include <string>

#include "OpenCL.h"

using namespace std;

template <typename T>
class CLAlgorithm
{
public:
    CLAlgorithm() {};
    virtual ~CLAlgorithm() {};

    void setContext(Context* context)
    {
        this->context = context;
    }

    void setCommandQueue(CommandQueue* queue)
    {
        this->queue = queue;
    }

    virtual const vector<size_t> getSupportedWorkGroupSizes() const {
        size_t maxWorkGroupSize = context->getInfo<size_t>(CL_DEVICE_MAX_WORK_GROUP_SIZE);

        maxWorkGroupSize = rootPowerOfTwo((unsigned int)maxWorkGroupSize, getWorkDimensions());

        vector<size_t> sizes;
        for(size_t i = 1; i <= maxWorkGroupSize; i <<= 1)
            sizes.push_back(i);

        return sizes;
    }

    virtual const size_t getOptimalWorkGroupSize() const {
        return getSupportedWorkGroupSizes().back();
    }

    virtual const string getName() = 0;
    virtual void init() = 0;
    virtual void upload(size_t workGroupSize, T* data, size_t size) = 0;
    virtual void run(size_t workGroupSize, size_t size) = 0;
    virtual void download(T* result, size_t size) = 0;
    virtual void cleanup() = 0;

    virtual const cl_uint getWorkDimensions() const {
        return 1;
    }

protected:
    Context* context;
    CommandQueue* queue;
};
