#ifndef SORTINGALGORITHM_H
#define SORTINGALGORITHM_H

#include <string>
#include <vector>
#include <typeinfo>
#include <iostream>

#include "Timer.h"

using namespace std;

/**
 * Abstract class to test sorting algorithms.
 */
template<typename T, size_t count>
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

        typedef T value_type;

        /**
         * This method runs the sorting algorithm on test data and prints results.
         */
        void runTest()
        {
            cout << "###############################################################################" << endl;
            cout << "# " << name << endl;
            cout << "#  Sorting " << count << " elements of type " << typeid(T).name() << " (" << ((sizeof(T) * count) >> 20) << " MiB)" << endl;

            // generate random array
            data = new T[count];
            for(size_t i = 0; i < count; i++)
                data[i] = rand();

            runStages();

            delete[] data;

            cout << "###############################################################################" << endl;
            cout << endl;
        }

        bool isSorted()
        {
            bool sorted = true;
            for(size_t i = 0; i < count - 1; i++)
                if(data[i] > data[i + 1])
                {
                    sorted = false;
                    break;
                }

            return sorted;
        }

    protected:
        virtual void runStages() = 0;

        T* data;
        Timer timer;
    private:
        string name;
};

#endif // SORTINGALGORITHM_H
