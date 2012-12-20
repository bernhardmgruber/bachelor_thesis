#ifndef GPUCLPPSCAN_H
#define GPUCLPPSCAN_H

#include "../../ScanAlgorithm.h"
#include "../../../common/GPUAlgorithm.h"

#include "clpp/clppScan_GPU.h"

namespace gpu
{
    namespace clpp
    {
        template<typename T>
        class Scan : public GPUAlgorithm<T>, public ScanAlgorithm
        {
            public:
                const string getName() override
                {
                    return "Scan (clpp) (inclusiv)";
                }

                bool isInclusiv() override
                {
                    return true;
                }

                void init(Context* context) override
                {
                    clppProgram::setBasePath("../common/libs/clpp/clpp/");
                    clppcontext.setup(0, 0);

                    s = new clppScan_GPU(&clppcontext, sizeof(T), 0);
                    assert(s->_context->clQueue != 0);
                }

                void upload(Context* context, CommandQueue* queue, size_t workGroupSize, T* data, size_t size) override
                {
                    assert(s->_context->clQueue != 0);
                    buffer = new T[size];
                    memcpy(buffer, data, sizeof(T) * size);
                    s->pushDatas(buffer, size);
                }

                void run(CommandQueue* queue, size_t workGroupSize, size_t size) override
                {
                    s->scan();
                }

                void download(CommandQueue* queue, T* result, size_t size) override
                {
                    s->popDatas();
                    memcpy(buffer, result, size * sizeof(T));
                }

                void cleanup() override
                {
                    delete s;
                    delete buffer;
                }

                virtual ~Scan() {}

            private:
                T* buffer;
                clppScan* s;
                clppContext clppcontext;
        };
    }
}

#endif // GPUCLPPSCAN_H
