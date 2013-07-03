#pragma once

#include "../../ScanAlgorithm.h"
#include "../../../common/CLAlgorithm.h"

#include "clpp/clppScan_GPU.h"

namespace gpu
{
    namespace clpp
    {
        template<typename T>
        class Scan : public CLAlgorithm<T>, public ScanAlgorithm
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

                void init() override
                {
                    clppProgram::setBasePath("../common/libs/clpp/clpp/");
                    clppcontext.setup(0, 0);

                    s = new clppScan_GPU(&clppcontext, sizeof(T), 0);
                    assert(s->_context->clQueue != 0);
                }

                void upload(size_t workGroupSize, T* data, size_t size) override
                {
                    assert(s->_context->clQueue != 0);
                    buffer = new T[size];
                    memcpy(buffer, data, sizeof(T) * size);
                    s->pushDatas(buffer, size);
                }

                void run(size_t workGroupSize, size_t size) override
                {
                    s->scan();
                }

                void download(T* result, size_t size) override
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
