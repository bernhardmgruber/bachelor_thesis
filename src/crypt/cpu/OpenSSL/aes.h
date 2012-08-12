#ifndef CPUOPENSSLAES_H
#define CPUOPENSSLAES_H

#include "../../CPUCryptAlgorithm.h"

#include <openssl/evp.h>
#include <openssl/aes.h>

using namespace std;

namespace cpu
{
    namespace openssl
    {
        /**
         * From: http://saju.net.in/code/misc/openssl_aes.c.txt
         */
        template<size_t count>
        class AES : public CPUCryptoAlgorithm<count>
        {
                using Base = CryptoAlgorithm<count>;

            public:
                AES()
                    : CPUCryptoAlgorithm<count>("AES (OpenSSL)")
                {
                }

            protected:

                byte key[32] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
                                0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F};

                EVP_CIPHER_CTX encryptContext;
                EVP_CIPHER_CTX decryptContext;

                void init()
                {
                    EVP_CIPHER_CTX_init(&encryptContext);
                    EVP_EncryptInit_ex(&encryptContext, EVP_aes_256_cbc(), NULL, key, nullptr);
                    EVP_CIPHER_CTX_init(&decryptContext);
                    EVP_DecryptInit_ex(&decryptContext, EVP_aes_256_cbc(), NULL, key, nullptr);
                }

                byte* ciphertext;
                int cipherLength;

                void encrypt(const byte* const src)
                {
                    /* max ciphertext len for a n bytes of plaintext is n + AES_BLOCK_SIZE -1 bytes */
                    int bufferLength = count + AES_BLOCK_SIZE;
                    ciphertext = new byte[bufferLength];

                    /* allows reusing of 'encryptContext' for multiple encryption cycles */
                    EVP_EncryptInit_ex(&encryptContext, nullptr, nullptr, nullptr, nullptr);

                    /* update ciphertext, c_len is filled with the length of ciphertext generated,
                      *len is the size of plaintext in bytes */
                    int len = count;
                    EVP_EncryptUpdate(&encryptContext, ciphertext, &bufferLength, src, len);

                    /* update ciphertext with the final remaining bytes */
                    int finalLength = 0;
                    EVP_EncryptFinal_ex(&encryptContext, ciphertext + bufferLength, &finalLength);

                    cipherLength = bufferLength + finalLength;
                }

                void decrypt(byte* const dest)
                {
                    /* plaintext will always be equal to or lesser than length of ciphertext*/
                    int p_len = cipherLength;
                    int f_len = 0;
                    byte* plaintext = new byte[p_len];

                    EVP_DecryptInit_ex(&decryptContext, nullptr, nullptr, nullptr, nullptr);
                    EVP_DecryptUpdate(&decryptContext, plaintext, &p_len, ciphertext, cipherLength);
                    EVP_DecryptFinal_ex(&decryptContext, plaintext + p_len, &f_len);

                    //*len = p_len + f_len;

                    memcpy(dest, plaintext, count * sizeof(byte));
                    delete[] plaintext;

                    //return plaintext;
                }

                void cleanup()
                {
                    EVP_CIPHER_CTX_cleanup(&encryptContext);
                    EVP_CIPHER_CTX_cleanup(&decryptContext);
                }
        };
    }
}

#endif // CPUOPENSSLAES_H
