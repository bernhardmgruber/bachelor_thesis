#ifndef CRYPTALGORITHM_H
#define CRYPTALGORITHM_H

#include <string>
#include <vector>
#include <typeinfo>
#include <iostream>

#include "Timer.h"

using namespace std;

typedef unsigned char byte;

/**
 * Abstract class to test crypto algorithms.
 */
template<size_t count>
class CryptoAlgorithm
{
    public:
        CryptoAlgorithm(string name)
            : name(name)
        {
        }

        virtual ~CryptoAlgorithm()
        {
        }

        /**
         * This method runs the crypto algorithm on test data and prints results.
         */
        void runTest()
        {
            cout << "###############################################################################" << endl;
            cout << "# " << name << endl;
            cout << "#  Processing " << ((sizeof(byte) * count) >> 10) << " KiB" << endl;

            // generate random array
            data = new byte[count];
            for(size_t i = 0; i < count; i++)
                data[i] = rand();

            buffer = new byte[count];

            runStages(data, buffer);

            delete[] data;
            delete[] buffer;

            cout << "###############################################################################" << endl;
            cout << endl;
        }

        bool verify()
        {
            return !memcmp(data, buffer, count * sizeof(byte));
        }

    protected:
        virtual void runStages(const byte* const src, byte* const dest) = 0;

        byte* data;
        Timer timer;
    private:
        byte* buffer;
        string name;
};

#endif // CRYPTALGORITHM_H
