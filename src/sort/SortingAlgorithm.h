#ifndef SORTINGALGORITHM_H
#define SORTINGALGORITHM_H

#include <string>
#include <vector>
#include <typeinfo>

#include "Timer.h"

using namespace std;

/**
 * Abstract class to test sorting algorithms.
 */
template<typename T, size_t size>
class SortingAlgorithm
{
    public:
        SortingAlgorithm(string name)
            : name(name)
        {

        }

        virtual ~SortingAlgorithm()
        {

        }

        /**
         * This method runs the sorting algorithm on test data and prints results.
         *
         * @return Returns true on success.
         */
        bool runTest()
        {
            timer.start();
            if(!init())
                return false;
            double initTime = timer.stop();

            timer.start();
            sort();
            double sortTime = timer.stop();

            timer.start();
            cleanup();
            double cleanupTime = timer.stop();

            // print results
            cout << "###############################################################################" << endl;
            cout << "# " << name << endl;
            cout << "# Sorting " << size << " elements of type " << typeid(T).name() << " (" << ((sizeof(T) * size) >> 20) << " MiB)" << endl;
            cout << "# Init    " << fixed << initTime << "s" << endl;
            cout << "# Sort    " << fixed << sortTime << "s" << endl;
            cout << "# Cleanup " << fixed << cleanupTime << "s" << endl;
            cout << "###############################################################################" << endl;
            cout << endl;

            return true;
        }

    protected:
        virtual bool init() = 0;
        virtual void sort() = 0;
        virtual void cleanup() = 0;

    private:
        Timer timer;
        string name;
};

#endif // SORTINGALGORITHM_H
