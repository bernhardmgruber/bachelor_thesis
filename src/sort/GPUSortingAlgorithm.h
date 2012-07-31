#ifndef OPENCLSORTINGALGORITHM_H
#define OPENCLSORTINGALGORITHM_H

#include <CL/CL.h>

#include "SortingAlgorithm.h"

template<typename T, size_t count>
class GPUSortingAlgorithm : public SortingAlgorithm<T, count>
{
    using Base = SortingAlgorithm<T, count>;

 public:
        GPUSortingAlgorithm(string name, Context* context, CommandQueue* queue)
            : SortingAlgorithm<T, count>(name), context(context), queue(queue)
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
            Base::timer.start();
            sort();
            double sortTime = Base::timer.stop();

            // download data
            Base::timer.start();
            download();
            double downloadTime = Base::timer.stop();

            // cleanup
            Base::timer.start();
            cleanup();
            double cleanupTime = Base::timer.stop();

            cout << "#  Init      " << fixed << initTime << "s" << endl;
            cout << "#  Upload    " << fixed << uploadTime << "s" << endl;
            cout << "#  Sort      " << fixed << sortTime << "s" << endl;
            cout << "#  Download  " << fixed << downloadTime << "s" << endl;
            cout << "#  Cleanup   " << fixed << cleanupTime << "s" << endl;
            cout << "#  " << (Base::isSorted() ? "SUCCESS" : "FAILED ") << "   " << fixed << (initTime + uploadTime + sortTime + downloadTime + cleanupTime) << "s" << endl;
        }

    protected:
        virtual bool init() = 0;
        virtual void upload() = 0;
        virtual void sort() = 0;
        virtual void download() = 0;
        virtual void cleanup() = 0;

        Context* context;
        CommandQueue* queue;
};

#endif // OPENCLSORTINGALGORITHM_H
