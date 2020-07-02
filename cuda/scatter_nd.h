#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

template <typename T>
void scatterNd_kernel_launcher(const T *src_ptr, const int *indices_ptr, T *dst_ptr,
                               const int src_rows, const int dst_rows, const int cols, cudaStream_t stream);