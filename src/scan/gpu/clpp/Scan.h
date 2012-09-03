#ifndef GPUCLPPSCAN_H
#define GPUCLPPSCAN_H

// In order to test that no value has been loosed ! Can take time to check !
#define PARAM_CHECK_HASLOOSEDVALUES 0
#define PARAM_BENCHMARK_LOOPS 20

// The number of bits to sort
#define PARAM_SORT_BITS 32

#include "../../GPUScanAlgorithm.h"

#include "clpp/clppScan_GPU.h"

using namespace std;

namespace gpu
{
    namespace clpp
    {
        template<typename T, size_t count>
        class Scan : public GPUScanAlgorithm<T, count>
        {
                using Base = GPUScanAlgorithm<T, count>;

            public:
                Scan(Context* context, CommandQueue* queue)
                    : GPUScanAlgorithm<T, count>("Prefix Sum (clpp)", context, queue)
                {
                    clppProgram::setBasePath("../common/libs/clpp/clpp/");

                    clppcontext.setup(0, 0);
                }

                virtual ~Scan()
                {
                }

            protected:
                clppScan* s;
                clppContext clppcontext;

                bool init()
                {
                    s = new clppScan_GPU(&clppcontext, sizeof(T), count);

                    assert(s->_context->clQueue != 0);

                    return true;
                }

                void upload()
                {
                    assert(s->_context->clQueue != 0);
                    s->pushDatas(ScanAlgorithm<T, count>::data, count);
                }

                void scan(size_t workGroupSize)
                {
                    s->scan();
                    s->waitCompletion();
                }

                void download()
                {
                    s->popDatas();
                }

                void cleanup()
                {
                    delete s;
                }
        };
    }
}

#endif // GPUCLPPSCAN_H
