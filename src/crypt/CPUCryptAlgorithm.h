#ifndef CPUCRYPTOALGORITHM_H
#define CPUCRYPTOALGORITHM_H

#include "CryptAlgorithm.h"

template<size_t count>
class CPUCryptoAlgorithm : public CryptoAlgorithm<count>
{
        using Base = CryptoAlgorithm<count>;

    public:
        CPUCryptoAlgorithm(string name)
            : CryptoAlgorithm<count>(name)
        {
        }

        virtual ~CPUCryptoAlgorithm()
        {
        }

        void runStages(const byte* const src, byte* const dest)
        {
            Base::timer.start();
            init();
            double initTime = Base::timer.stop();

            Base::timer.start();
            encrypt(src);
            double encryptTime = Base::timer.stop();

            Base::timer.start();
            decrypt(dest);
            double decryptTime = Base::timer.stop();

            cout << "#  Init      " << fixed << initTime << "s" << flush << endl;
            cout << "#  Encrypt   " << fixed << encryptTime << "s" << flush << endl;
            cout << "#  Decrypt   " << fixed << decryptTime << "s" << flush << endl;
            cout << "#  " << (Base::verify() ? "SUCCESS" : "FAILED ") << "   " << fixed << (initTime + encryptTime + decryptTime) << "s" << flush << endl;
        }

    protected:
        virtual void init() = 0;
        virtual void encrypt(const byte* const src) = 0;
        virtual void decrypt(byte* const dest) = 0;
};

template <template <size_t> class T, size_t count>
void runCPU()
{
    CryptoAlgorithm<count>* alg;
    alg = new T<count>();
    alg->runTest();
    delete alg;
}

#define RUN(algorithmTestClass, count) runCPU<algorithmTestClass, count>();

#endif // CPUCRYPTOALGORITHM_H
