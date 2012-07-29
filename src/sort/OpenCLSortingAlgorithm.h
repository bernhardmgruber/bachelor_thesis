#ifndef OPENCLSORTINGALGORITHM_H
#define OPENCLSORTINGALGORITHM_H

#include <CL/CL.h>

#include "SortingAlgorithm.h"

class OpenCLSortingAlgorithm : public SortingAlgorithm
{
 public:
        OpenCLSortingAlgorithm(string name, cl_context context)
            : SortingAlgorithm(name), context(context)
        {
        }

        virtual ~OpenCLSortingAlgorithm()
        {
        }

    protected:
        virtual bool init() = 0;
        virtual void sort() = 0;
        virtual void cleanup() = 0;

    private:
        cl_context context;
};

#endif // OPENCLSORTINGALGORITHM_H
