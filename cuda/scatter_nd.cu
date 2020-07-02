#include "cuda/scatter_nd.h"
#include <assert.h>
#include <cmath>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <sys/time.h>
using namespace std;

template <typename T>
__global__ void scatterNd_kernel(const T *src, const int *indices, T *dst,
                                 const int cols) {
  int tgt_row = __ldg(&indices[blockIdx.x]);
  int src_row = blockIdx.x;

  int src_row_offset = src_row * cols;
  int tgt_row_offset = tgt_row * cols;

  for (int tid = threadIdx.x; tid < cols; tid += blockDim.x)
    dst[tgt_row_offset + tid] = __ldg(&src[src_row_offset + tid]);
}

template <typename T>
void scatterNd_kernel_launcher(const T *src_ptr, const int *indices_ptr,
                               T *dst_ptr, const int src_rows,
                               const int dst_rows, const int cols,
                               cudaStream_t stream) {
  dim3 grid(src_rows);
  dim3 block(cols);

  if (block.x > 1024)
    block.x = 1024;

  scatterNd_kernel<<<grid, block, 0, stream>>>(src_ptr, indices_ptr, dst_ptr,
                                               cols);
}

template void scatterNd_kernel_launcher(const float *src_ptr,
                                        const int *indices_ptr, float *dst_ptr,
                                        const int src_rows, const int dst_rows,
                                        const int cols, cudaStream_t stream);

template void scatterNd_kernel_launcher(const __half *src_ptr,
                                        const int *indices_ptr, __half *dst_ptr,
                                        const int src_rows, const int dst_rows,
                                        const int cols, cudaStream_t stream);
