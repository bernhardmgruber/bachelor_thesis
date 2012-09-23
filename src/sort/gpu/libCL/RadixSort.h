// Copyright [2011] [Geist Software Labs Inc.]
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef LibCLRadixSort
#define LibCLRadixSort

#include "../../../common/libs/libcl/oclContext.h"
#include "../../../common/libs/libcl/oclBuffer.h"
#include "../../../common/libs/libcl/sort/oclRadixSort.h"

#include "../../../common/GPUAlgorithm.h"
#include "../../SortAlgorithm.h"

namespace gpu
{
    namespace libcl
    {
        /**
         * From: http://www.libcl.org/
         * Modified by Bernhard Manfred Gruber to be key only.
         */
        template<typename T, size_t count>
        class RadixSort : public GPUAlgorithm<T, count>, public SortAlgorithm
        {
            public:
                string getName() override
                {
                    return "Radix sort (LibCL)";
                }

                bool isInPlace() override
                {
                    return false;
                }

                void init(Context* context) override
                {
                    ctx = oclContext::create(oclContext::VENDOR_NVIDIA, CL_DEVICE_TYPE_GPU);
                    if (!ctx)
                        ctx = oclContext::create(oclContext::VENDOR_AMD, CL_DEVICE_TYPE_GPU);

                    if (!ctx)
                        throw OpenCLException("no OpenCL capable platform detected");

                    program = new oclRadixSort(*ctx);
                    if (!program->compile())
                        throw OpenCLException("compilation failed!");
                }

                void upload(Context* context, CommandQueue* queue, size_t workGroupSize, T* data) override
                {
                    bfKey = new oclBuffer(*ctx, "bfKey");
                    bfVal = new oclBuffer(*ctx, "bfVal");

                    bfKey->create<cl_uint> (CL_MEM_READ_WRITE, count);
                    bfVal->create<cl_uint> (CL_MEM_READ_WRITE, count);

                    if (!bfKey->map(CL_MAP_READ | CL_MAP_WRITE))
                        throw OpenCLException("map failed!");
                    //if (!bfVal.map(CL_MAP_READ | CL_MAP_WRITE))
                    //    throw OpenCLException("map failed!");

                    T* keyPtr = bfKey->ptr<T>();
                    //T* ptrVal = bfVal.ptr<T>();
                    memcpy(keyPtr, data, sizeof(T) * count);

                    bfVal->write();
                    //bfKey.write();
                }

                void run(CommandQueue* queue, size_t workGroupSize) override
                {
                    program->compute(ctx->getDevice(0), *bfKey, *bfVal, 0, 32);
                }

                void download(CommandQueue* queue, T* result) override
                {
                    bfVal->read();
                    //bfKey.read();

                    memcpy(result, keyPtr, sizeof(T) * count);

                    bfVal->unmap();
                    //bfKey.unmap();
                }

                void cleanup() override
                {
                    delete program;
                    delete ctx;
                    delete bfKey;
                    delete bfVal;
                }

                virtual ~RadixSort() {}

            private:
                oclContext* ctx;
                oclRadixSort* program;
                oclBuffer* bfKey;
                oclBuffer* bfVal;

                T* keyPtr;
        };
    }
}

#endif // LibCLRadixSort
