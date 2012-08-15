#ifndef SORTINGALGORITHM_H
#define SORTINGALGORITHM_H

#include <string>
#include <vector>
#include <typeinfo>
#include <iostream>

#include "../common/Timer.h"

using namespace std;

#define DEFAULT_DATA_TYPE int
#define DEFAULT_DATA_COUNT 256 * 256

/**
 * Abstract class to test sorting algorithms.
 */
template<typename T = DEFAULT_DATA_TYPE, size_t count = DEFAULT_DATA_COUNT>
class ScanAlgorithm
{
    public:
        ScanAlgorithm(string name)
            : name(name)
        {
        }

        virtual ~ScanAlgorithm()
        {
        }

        /**
         * This method runs the scan algorithm on test data and prints results.
         */
        void runTest()
        {
            cout << "###############################################################################" << endl;
            cout << "# " << name << endl;
            cout << "#  Scanning " << count << " elements of type " << typeid(T).name() << " (" << ((sizeof(T) * count) >> 10) << " KiB)" << endl;

            // generate random array
            data = new T[count];
            for(size_t i = 0; i < count; i++)
                data[i] = rand();

            scanResult = new T[count];

            runStages();

            delete[] data;
            delete[] scanResult;

            cout << "###############################################################################" << endl;
            cout << endl;
        }

        bool verify()
        {
            if(data[0] != scanResult[0])
                return false;

            for(size_t i = 1; i < count; i++)
                if(scanResult[i] !=  scanResult[i - 1] + data[i])
                    return false;

            return true;
        }

    protected:
        virtual void runStages() = 0;

        T* data;
        T* scanResult;
        Timer timer;

    private:
        string name;
};

#endif // SORTINGALGORITHM_H
