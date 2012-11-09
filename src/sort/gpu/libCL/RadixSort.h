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

#include "../../../common/libs/libCL/oclContext.h"
#include "../../../common/libs/libCL/oclBuffer.h"
#include "../../../common/libs/libCL/sort/oclRadixSort.h"

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
        template<typename T>
        class RadixSort : public GPUAlgorithm<T>, public SortAlgorithm
        {
            public:
                const string getName() override
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

                void upload(Context* context, CommandQueue* queue, size_t workGroupSize, T* data, size_t size) override
                {
                    bfKey = new oclBuffer(*ctx, "bfKey");
                    bfVal = new oclBuffer(*ctx, "bfVal");

                    bfKey->create<cl_uint> (CL_MEM_READ_WRITE, size);
                    bfVal->create<cl_uint> (CL_MEM_READ_WRITE, size);

                    if (!bfKey->map(CL_MAP_READ | CL_MAP_WRITE))
                        throw OpenCLException("map failed!");
                    //if (!bfVal.map(CL_MAP_READ | CL_MAP_WRITE))
                    //    throw OpenCLException("map failed!");

                    T* keyPtr = bfKey->ptr<T>();
                    //T* ptrVal = bfVal.ptr<T>();
                    memcpy(keyPtr, data, sizeof(T) * size);

                    bfVal->write();
                    //bfKey.write();
                }

                void run(CommandQueue* queue, size_t workGroupSize, size_t size) override
                {
                    program->compute(ctx->getDevice(0), *bfKey, *bfVal, 0, 32);
                }

                void download(CommandQueue* queue, T* result, size_t size) override
                {
                    bfVal->read();
                    //bfKey.read();

                    memcpy(result, keyPtr, sizeof(T) * size);

                    bfVal->unmap();
                    //bfKey.unmap();

                    delete bfKey;
                    delete bfVal;
                }

                void cleanup() override
                {
                    delete program;
                    delete ctx;
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
