#ifndef GPUCLPPSCAN_H
#define GPUCLPPSCAN_H

#include "../../GPUScanAlgorithm.h"

#include "clpp/clppScan_GPU.h"

namespace gpu
{
    namespace clpp
    {
        template<typename T, size_t count>
        class Scan : public GPUScanAlgorithm<T, count>
        {
            public:
                string getName() override
                {
                    return "Scan (clpp)";
                }

                void init(Context* context) override
                {
                    clppProgram::setBasePath("../common/libs/clpp/clpp/");
                    clppcontext.setup(0, 0);

                    s = new clppScan_GPU(&clppcontext, sizeof(T), count);
                    assert(s->_context->clQueue != 0);
                }

                void upload(Context* context, T* data) override
                {
                    assert(s->_context->clQueue != 0);
                    buffer = new T[count];
                    memcpy(buffer, data, sizeof(T) * count);
                    s->pushDatas(buffer, count);
                }

                void scan(CommandQueue* queue, size_t workGroupSize) override
                {
                    s->scan();
                    s->waitCompletion();
                }

                void download(CommandQueue* queue, T* result) override
                {
                    s->popDatas();
                    memcpy(buffer, result, count * sizeof(T));
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
