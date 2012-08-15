#ifndef CLPPSORT_H
#define CLPPSORT_H

void* clpp_init(size_t size);
void clpp_upload(void* s, int* data, size_t size);
void clpp_sort(void* s);
void clpp_download(void* s);
void clpp_cleanup(void* s);

#endif // CLPPSORT_H
