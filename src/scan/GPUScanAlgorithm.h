#ifndef GPUSCANALGORITHM_H
#define GPUSCANALGORITHM_H

#include <CL/CL.h>

#include <map>
#include <algorithm>

#include "ScanAlgorithm.h"

template<typename T, size_t count>
class GPUScanAlgorithm : public ScanAlgorithm<T, count>
{
    using Base = ScanAlgorithm<T, count>;

 public:
        GPUScanAlgorithm(string name, Context* context, CommandQueue* queue, bool useMultipleWorkGroupSizes = false)
            : ScanAlgorithm<T, count>(name), context(context), queue(queue), useMultipleWorkGroupSizes(useMultipleWorkGroupSizes)
        {
        }

        virtual ~GPUScanAlgorithm()
        {
        }

        void runStages()
        {
            // run custom initialization
            Base::timer.start();
            init();
            double initTime = Base::timer.stop();

            // upload data
            Base::timer.start();
            upload();
            double uploadTime = Base::timer.stop();

            // run scan algorithm
            size_t maxWorkGroupSize = min(context->getInfoSize(CL_DEVICE_MAX_WORK_GROUP_SIZE), count);
            map<int, double> scanTimes;
            if(!useMultipleWorkGroupSizes)
            {
                Base::timer.start();
                scan(maxWorkGroupSize);
                scanTimes[maxWorkGroupSize] = Base::timer.stop();
            }
            else
            {
                for(size_t i = 1; i <= maxWorkGroupSize; i <<= 1)
                {
                    // check if work group size divides the input
                    if(count % i == 0)
                    {
                        Base::timer.start();
                        scan(i);
                        scanTimes[i] = Base::timer.stop();
                    }
                }
            }

            // download data
            Base::timer.start();
            download();
            double downloadTime = Base::timer.stop();

            // cleanup
            Base::timer.start();
            cleanup();
            double cleanupTime = Base::timer.stop();

            cout << "#  Init      " << fixed << initTime << "s" << flush << endl;
            cout << "#  Upload    " << fixed << uploadTime << "s" << flush << endl;

            for(auto entry : scanTimes)
                cout << "#  Sort      " << fixed << entry.second << "s " << "( WG size: " << entry.first << ")" << flush << endl;

            cout << "#  Download  " << fixed << downloadTime << "s" << flush << endl;
            cout << "#  Cleanup   " << fixed << cleanupTime << "s" << flush << endl;
            cout << "#  " << (Base::verify() ? "SUCCESS" : "FAILED ") << "   " << fixed << (initTime + uploadTime + min_element(scanTimes.begin(), scanTimes.end(), [](pair<int, double> a, pair<int, double> b) { return a.second < b.second; })->second + downloadTime + cleanupTime) << "s (fastest)" << flush << endl;
        }

    protected:
        virtual bool init() = 0;
        virtual void upload() = 0;
        virtual void scan(size_t workGroupSize) = 0;
        virtual void download() = 0;
        virtual void cleanup() = 0;

        Context* context;
        CommandQueue* queue;
        bool useMultipleWorkGroupSizes;
};

template <template <typename, size_t> class T, size_t count, typename V>
void runGPU(Context* context, CommandQueue* queue)
{
    ScanAlgorithm<V, count>* alg;
    alg = new T<V, count>(context, queue);
    alg->runTest();
    delete alg;
}

#define RUN_CL(algorithmTestClass, count, valueType) runGPU<algorithmTestClass, count, valueType>(context, queue);

#endif // GPUSCANALGORITHM_H
