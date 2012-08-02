#ifndef OPENCLSORTINGALGORITHM_H
#define OPENCLSORTINGALGORITHM_H

#include <CL/CL.h>
#include <map>

#include "SortingAlgorithm.h"

template<typename T, size_t count>
class GPUSortingAlgorithm : public SortingAlgorithm<T, count>
{
    using Base = SortingAlgorithm<T, count>;

 public:
        GPUSortingAlgorithm(string name, Context* context, CommandQueue* queue, bool useMultipleWorkGroupSizes = false)
            : SortingAlgorithm<T, count>(name), context(context), queue(queue), useMultipleWorkGroupSizes(useMultipleWorkGroupSizes)
        {
        }

        virtual ~GPUSortingAlgorithm()
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

            // run sorting algorithm
            size_t maxWorkGroupSize = min(context->getInfoSize(CL_DEVICE_MAX_WORK_GROUP_SIZE), count);
            map<int, double> sortTimes;
            if(!useMultipleWorkGroupSizes)
            {
                Base::timer.start();
                sort(maxWorkGroupSize);
                sortTimes[maxWorkGroupSize] = Base::timer.stop();
            }
            else
            {
                for(size_t i = 1; i <= maxWorkGroupSize; i <<= 1)
                {
                    // check if work group size divides the input
                    if(count % i == 0)
                    {
                        Base::timer.start();
                        sort(i);
                        sortTimes[i] = Base::timer.stop();
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

            for(auto entry : sortTimes)
                cout << "#  Sort      " << fixed << entry.second << "s " << "( WG size: " << entry.first << ")" << flush << endl;

            cout << "#  Download  " << fixed << downloadTime << "s" << flush << endl;
            cout << "#  Cleanup   " << fixed << cleanupTime << "s" << flush << endl;
            cout << "#  " << (Base::isSorted() ? "SUCCESS" : "FAILED ") << "   " << fixed << (initTime + uploadTime + min_element(sortTimes.begin(), sortTimes.end(), [](pair<int, double> a, pair<int, double> b) { return a.second < b.second; })->second + downloadTime + cleanupTime) << "s (fastest)" << flush << endl;
        }

    protected:
        void sort()
        {
        }

        virtual bool init() = 0;
        virtual void upload() = 0;
        virtual void sort(size_t workGroupSize) = 0;
        virtual void download() = 0;
        virtual void cleanup() = 0;

        Context* context;
        CommandQueue* queue;
        bool useMultipleWorkGroupSizes;
};

#endif // OPENCLSORTINGALGORITHM_H
