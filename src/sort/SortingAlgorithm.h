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
         *
         * @return Returns true on success.
         */
        bool runTest()
        {
            cout << "###############################################################################" << endl;
            cout << "# " << name << endl;
            cout << "#  Sorting " << count << " elements of type " << typeid(T).name() << " (" << ((sizeof(T) * count) >> 20) << " MiB)" << endl;

            // generate random array
            data = new T[count];
            for(size_t i = 0; i < count; i++)
                data[i] = rand();

            // run custom initialization
            bool initSuccessful = true;
            timer.start();
            if(!init())
                initSuccessful = false;
            double initTime = timer.stop();

            if(initSuccessful)
            {
                // run sorting algorithm
                timer.start();
                sort();
                double sortTime = timer.stop();

                // cleanup
                timer.start();
                cleanup();
                double cleanupTime = timer.stop();

                // verify
                bool sorted = true;
                for(size_t i = 0; i < count - 1; i++)
                    if(data[i] > data[i + 1])
                    {
                        sorted = false;
                        break;
                    }

                cout << "#  Init    " << fixed << initTime << "s" << endl;
                cout << "#  Sort    " << fixed << sortTime << "s" << endl;
                cout << "#  Cleanup " << fixed << cleanupTime << "s" << endl;
                cout << "#  " << (sorted ? "SUCCESS" : "FAILED") << " " << fixed << (initTime + sortTime + cleanupTime) << "s" << endl;
            }
            else
            {
                cout << "#  Initialization FAILED" << endl;
            }

            cout << "###############################################################################" << endl;
            cout << endl;

            return true;
        }

    protected:
        T* data;

        virtual bool init() = 0;
        virtual void sort() = 0;
        virtual void cleanup() = 0;

    private:
        Timer timer;
        string name;
};

template <typename T>
inline void swap(T& a, T& b)
{
    T tmp = a;
    a = b;
    b = tmp;
}

#endif // SORTINGALGORITHM_H
