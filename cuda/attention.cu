/*
 * Copyright (C) 2020 ByteDance Inc
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include "cuda/attention.h"

namespace effectivetransformer {
namespace cuda {

// Reduce code comes from Nvidia's DeepLearningExamples
// https://github.com/NVIDIA/DeepLearningExamples/blob/master/FasterTransformer/v1/fastertransformer/cuda/open_attention.cu#L29-L101

/**
 * Multi-head attetion open sourced
 */

#define FINAL_MASK 0xffffffff

template <typename T>
__inline__ __device__
T warpReduceSum(T val)
{
  #pragma unroll
  for(int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}

/* Calculate the sum of all elements in a block */
template <typename T>
  __inline__ __device__
T blockReduceSum(T val)
{
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum<T>(val);

  if(lane == 0)
    shared[wid] = val;

  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : (T)(0.0f);
  val = warpReduceSum<T>(val);

  return val;
}

template <typename T>
  __inline__ __device__
T warpReduceMax(T val)
{
  #pragma unroll
  for(int mask = 16; mask > 0; mask >>= 1)
    val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
  return val;
}

/* Calculate the maximum of all elements in a block */
template <typename T>
  __inline__ __device__
T blockReduceMax(T val)
{
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f; // in-warp idx
  int wid = threadIdx.x >> 5;  // warp idx

  val = warpReduceMax(val); // get maxx in each warp

  if(lane == 0) // record in-warp maxx by warp Idx
    shared[wid] = val;

  __syncthreads();


  val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : 0;
  val = warpReduceMax(val);

  return val;
}

__inline__ __device__
int target_index(int id1, int id2, int id3, int id4,
                 int dim_1, int dim_2, int dim_3, int dim_4)
{
  return id1 * (dim_2 * dim_3 * dim_4) +
      id3 * (dim_2 * dim_4) + id2 * dim_4 + id4;
}

/// ***************************** add bias & pad *****************************
template<typename T>
__global__
void add_QKV_bias_padding(
    T* Q, const T* bias_Q,
    T* K, const T* bias_K,
    T* V, const T* bias_V,
    T* q_buf_, T* k_buf_, T* v_buf_,
    const int batch_size, const int seq_len,
    const int head_num, const int size_per_head,
    const int* batch_idx, const int* word_idx)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_id  = batch_idx[blockIdx.x];
  int seq_id    = word_idx[blockIdx.x];
  int head_id   = (tid % (head_num * size_per_head)) / size_per_head;
  int id        = tid % size_per_head;
  int target_id = target_index(batch_id, seq_id, head_id, id,
                               batch_size, seq_len, head_num, size_per_head);
  int bias_id   = threadIdx.x;

  T* src_ptr = (T*)Q;
  T* dst_ptr = (T*)q_buf_;
  const T* bias_ptr = (const T*)bias_Q;
  dst_ptr[target_id] = src_ptr[tid] + __ldg(&bias_ptr[bias_id]);

  src_ptr = (T*)K;
  dst_ptr = (T*)k_buf_;
  bias_ptr = (const T*)bias_K;
  dst_ptr[target_id] = src_ptr[tid] + __ldg(&bias_ptr[bias_id]);

  src_ptr = (T*)V;
  dst_ptr = (T*)v_buf_;
  bias_ptr = (const T*)bias_V;
  dst_ptr[target_id] = src_ptr[tid] + __ldg(&bias_ptr[bias_id]);
}

template <>
__global__
void add_QKV_bias_padding(
    __half* Q, const __half* bias_Q,
    __half* K, const __half* bias_K,
    __half* V, const __half* bias_V,
    __half* q_buf_, __half* k_buf_, __half* v_buf_,
    const int batch_size, const int seq_len,
    const int head_num, const int size_per_head,
    const int* batch_idx, const int* word_idx)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_id  = batch_idx[blockIdx.x];
  int seq_id    = word_idx[blockIdx.x];
  int head_id   = (tid % (head_num * size_per_head)) / size_per_head;
  int id        = tid % size_per_head;
  int target_id = target_index(batch_id, seq_id, head_id, id,
                               batch_size, seq_len, head_num, size_per_head);
  int bias_id = threadIdx.x;

  half2* src_ptr = (half2*)Q;
  half2* dst_ptr = (half2*)q_buf_;
  const half2* bias_ptr = (const half2*)bias_Q;
  dst_ptr[target_id] = __hadd2(src_ptr[tid],  __ldg(&bias_ptr[bias_id]));

  src_ptr = (half2*)K;
  dst_ptr = (half2*)k_buf_;
  bias_ptr = (const half2*)bias_K;
  dst_ptr[target_id] = __hadd2(src_ptr[tid],  __ldg(&bias_ptr[bias_id]));

  src_ptr = (half2*)V;
  dst_ptr = (half2*)v_buf_;
  bias_ptr = (const half2*)bias_V;
  dst_ptr[target_id] = __hadd2(src_ptr[tid],  __ldg(&bias_ptr[bias_id]));
}

template<typename T>
void add_QKV_bias_padding_kernelLauncher(
    T* Q, const T* bias_Q,
    T* K, const T* bias_K,
    T* V, const T* bias_V,
    T* q_buf_, T* k_buf_, T* v_buf_,
    const int valid_word_num,
    const int batch_size, const int seq_len,
    const int head_num, const int size_per_head,
    const int* batch_idx, const int* word_idx,
    const cudaStream_t stream)
{
  dim3 grid;
  dim3 block;
  grid.x  = valid_word_num;
  block.x = head_num * size_per_head;

  add_QKV_bias_padding<float><<<grid, block, 0, stream>>>(
    Q, bias_Q, K, bias_K, V, bias_V, q_buf_, k_buf_, v_buf_,
    batch_size, seq_len, head_num, size_per_head, batch_idx, word_idx);
}

template<>
void add_QKV_bias_padding_kernelLauncher(
    __half* Q, const __half* bias_Q,
    __half* K, const __half* bias_K,
    __half* V, const __half* bias_V,
    __half* q_buf_, __half* k_buf_, __half* v_buf_,
    const int valid_word_num,
    const int batch_size, const int seq_len,
    const int head_num, const int size_per_head,
    const int* batch_idx, const int* word_idx,
    const cudaStream_t stream)
{
  dim3 grid;
  dim3 block;
  grid.x  = valid_word_num;
  block.x = head_num * size_per_head / 2;

  add_QKV_bias_padding<__half><<<grid, block, 0, stream>>>(
    Q, bias_Q, K, bias_K, V, bias_V, q_buf_, k_buf_, v_buf_,
    batch_size, seq_len, head_num, size_per_head / 2, batch_idx, word_idx);
}

template void add_QKV_bias_padding_kernelLauncher<__half>(
    __half* Q, const __half* bias_Q,
    __half* K, const __half* bias_K,
    __half* V, const __half* bias_V,
    __half* q_buf_, __half* k_buf_, __half* v_buf_,
    const int valid_word_num,
    const int batch_size, const int seq_len,
    const int head_num, const int size_per_head,
    const int* batch_idx, const int* word_idx,
    const cudaStream_t stream);

template void add_QKV_bias_padding_kernelLauncher<float>(
    float* Q, const float* bias_Q,
    float* K, const float* bias_K,
    float* V, const float* bias_V,
    float* q_buf_, float* k_buf_, float* v_buf_,
    const int valid_word_num,
    const int batch_size, const int seq_len,
    const int head_num, const int size_per_head,
    const int* batch_idx, const int* word_idx,
    const cudaStream_t stream);
/// *********************************** fin ***********************************


/// ************************** softmax for attention **************************
// softmax kernel code is copied from Nvidia's DeepLearningExamples :
// https://github.com/NVIDIA/DeepLearningExamples/blob/master/FasterTransformer/v1/fastertransformer/cuda/open_attention.cu#L189-L268
template <typename T>
__global__
void softmax_kernel(T* qk_buf_, const T* attr_mask,
  const int batch_size, const int head_num, const int seq_len, const T scaler)
{
  int batch_id = blockIdx.x / head_num;
  int qk_offset = blockIdx.x * seq_len * seq_len;
  int mask_offset = batch_id * seq_len * seq_len;

  __shared__ float s_sum, s_max;

  for(int i = 0; i < seq_len; ++i)
  {
    float qk = threadIdx.x < seq_len
                  ? (float)qk_buf_[threadIdx.x + qk_offset] : 0.0f;
    float mask_val = threadIdx.x < seq_len
                  ? (float)attr_mask[threadIdx.x + mask_offset] : 0.0f;

    mask_val = (1.0f - mask_val) * -10000.0f;

    float tmp = threadIdx.x < seq_len
                  ? (float)(qk * (float)scaler + mask_val): -1e-20f;

    float max_val = blockReduceMax<float>(tmp);

    if(threadIdx.x == 0)
      s_max = max_val;
    __syncthreads();

    qk = threadIdx.x < seq_len ? __expf(tmp - s_max) : 0.0f;

    float sum_val = blockReduceSum<float>(qk);

    if(threadIdx.x == 0)
    {
      s_sum = sum_val + 1e-6f;
    }
    __syncthreads();

    if(threadIdx.x < seq_len)
      qk_buf_[threadIdx.x + qk_offset] = (T)(qk / s_sum);

    qk_offset += seq_len;
    mask_offset += seq_len;
  }
}

template <typename T>
__global__
void softmax_kernel_v2(T* qk_buf_, const T* attr_mask,
  const int batch_size, const int head_num,
  const int seq_len, const float scaler)
{
  int batch_id = blockIdx.x / head_num / seq_len;
  int seq_id = blockIdx.x % seq_len;
  int qk_offset = blockIdx.x * seq_len;
  int mask_offset = batch_id * seq_len * seq_len + seq_id * seq_len;

  __shared__ float s_sum, s_max;

  float qk = threadIdx.x < seq_len
                ? (float)qk_buf_[threadIdx.x + qk_offset] : 0.0f;
  float mask_val = threadIdx.x < seq_len
                ? (float)attr_mask[threadIdx.x + mask_offset] : 0.0f;

  mask_val = (1.0f - mask_val) * -10000.0f;

  float tmp = threadIdx.x < seq_len
                ? (float)(qk * (float)scaler + mask_val) : -1e-20f;
  float max_val = blockReduceMax<float>(tmp);
  if(threadIdx.x == 0)
    s_max = max_val;
  __syncthreads();

  float qk_tmp = threadIdx.x < seq_len ? __expf((float)(tmp - s_max)) : 0.0f;
  float sum_val = blockReduceSum<float>(qk_tmp);

  if(threadIdx.x == 0)
  {
    s_sum = sum_val + 1e-6f;
  }
  __syncthreads();

  if(threadIdx.x < seq_len)
    qk_buf_[threadIdx.x + qk_offset] = (T)(qk_tmp / s_sum);
}

//grid = (seq_len/word_per_thread, batch_size, head_num)
//block.x = max(32, (seq_len + 31)/32*32)
template <typename T>
__global__
void softmax_kernel_v3(T* qk_buf_, const T* attr_mask,
  const int batch_size, const int head_num,
  const int seq_len, const T scalar)
{

  bool qual = threadIdx.x < seq_len;
  for (int seq_id = blockIdx.x ; seq_id < seq_len ; seq_id += gridDim.x){
    float tmp = -1e20f;
    int qk_offset;
    __shared__ float s_mean, s_max;
    if (qual){
      qk_offset = ((blockIdx.y*head_num + blockIdx.z)*seq_len + seq_id) *seq_len + threadIdx.x;
      int mask_offset = (blockIdx.y * seq_len + seq_id) * seq_len + threadIdx.x;

      float qk = static_cast<float>(qk_buf_[qk_offset]);
      float mask_val = static_cast<float>(__ldg(&attr_mask[mask_offset]));

      mask_val = (1.0f - mask_val) * -10000.0f;

      tmp = qk * static_cast<float>(scalar) + mask_val;
    }

    float max_val = blockReduceMax<float>(tmp);
    if (threadIdx.x == 0){
      s_max = max_val;
    }
    __syncthreads();

    float qk_tmp = qual ? __expf(tmp - s_max) : 0.0f;
    float sum_val = blockReduceSum<float>(qk_tmp);
    if (threadIdx.x == 0){
      s_mean = sum_val + 1e-6f;
      s_mean = __fdividef(1.0f, s_mean);
    }
    __syncthreads();

    if(qual)
      qk_buf_[qk_offset] = (T)(qk_tmp * s_mean);
  }
}


//grid = (seq_len/word_per_thread, batch_size, head_num)
//block.x = max(32, (seq_len/2 + 31)/32*32)
//seq_len % 2 == 0
template <>
__global__
void softmax_kernel_v3(half* qk_buf_, const half* attr_mask,
  const int batch_size, const int head_num,
  const int seq_len, const half scalar)
{
  int threadIdx2 = threadIdx.x << 1;
  bool qual = threadIdx2 < seq_len;
  half2* qk_buf_half2Ptr = (half2*) qk_buf_;
  const half2* attr_mask_half2Ptr = (const half2*) attr_mask;
  __shared__ float s_mean, s_max;
  for (int seq_id = blockIdx.x ; seq_id < seq_len ; seq_id += gridDim.x){
    int qk_offset;
    half2 tmp = __float2half2_rn(0.0f);

    float max_val = -1e20f;
    half2 qk;
    if (qual){
      qk_offset = ((((blockIdx.y*head_num + blockIdx.z)*seq_len + seq_id) *seq_len) >> 1) + threadIdx.x;
      int mask_offset = (((blockIdx.y * seq_len + seq_id) * seq_len) >> 1) + threadIdx.x;

      qk = qk_buf_half2Ptr[qk_offset];
      half2 mask_val = __ldg(&attr_mask_half2Ptr[mask_offset]);
      half2 mask_val_tmp = __hmul2(__hsub2(__float2half2_rn(1.0f), mask_val), __float2half2_rn(-10000.0f));
      tmp = __hadd2(__hmul2(__half2half2(scalar), qk), mask_val_tmp);
      max_val = fmax((float)tmp.x, (float)tmp.y);
    }

    max_val = blockDim.x <= 32 ? warpReduceMax(max_val) : blockReduceMax<float>(max_val);

    if (threadIdx.x == 0){
      s_max = max_val;
    }
    __syncthreads();

    if (qual){
      tmp = h2exp(__hsub2(tmp, __float2half2_rn(s_max)));
    }
    float sum_val = blockDim.x <= 32 ? warpReduceSum((float)(tmp.x + tmp.y)) : blockReduceSum<float>((float)(tmp.x + tmp.y));

    if (threadIdx.x == 0){
      s_mean = sum_val + 1e-6f;
      s_mean = __fdividef(1.0f, s_mean);
    }
    __syncthreads();

    if(qual){
      qk = __hmul2(tmp, __float2half2_rn(s_mean));
      qk_buf_half2Ptr[qk_offset] = qk;
    }
  }
}

template <typename T>
__global__
void softmax_kernel_v3_LE32(T* qk_buf_, const T* attr_mask,
  const int batch_size, const int head_num,
  const int seq_len, const T scalar)
{
  bool qual = threadIdx.x < seq_len;
  for (int seq_id = blockIdx.x ; seq_id < seq_len ; seq_id += gridDim.x){
    int qk_offset;
    __shared__ float s_mean, s_max;
    float tmp = -1e20f;
    if (qual){
      qk_offset = ((blockIdx.y*head_num + blockIdx.z)*seq_len + seq_id) *seq_len + threadIdx.x;
      int mask_offset = (blockIdx.y * seq_len + seq_id) * seq_len + threadIdx.x;

      float qk = static_cast<float>(qk_buf_[qk_offset]);
      float mask_val = static_cast<float>(__ldg(&attr_mask[mask_offset]));

      mask_val = (1.0f - mask_val) * -10000.0f;

      tmp = static_cast<float>(qk) * static_cast<float>(scalar) + mask_val;
    }
    float max_val = warpReduceMax<float>(tmp);

    if (threadIdx.x == 0){
      s_max = max_val;
    }
    __syncthreads();

    tmp = qual ? __expf(tmp - s_max) : 0.0f;
    float sum_val = warpReduceSum<float>(tmp);

    if (threadIdx.x == 0){
      s_mean = sum_val + 1e-6f;
      s_mean = __fdividef(1.0f, s_mean);
    }
    __syncthreads();

    if(qual)
      qk_buf_[qk_offset] = (T)(tmp * s_mean);
  }
}

template <typename T>
void softmax_kernel_kernelLauncher(
    T* qk_buf_, const T* attr_mask,
    const int batch_size, const int head_num, const int seq_len,
    const T scaler,
    const cudaStream_t stream) {
  dim3 grid;
  dim3 block;

  //deal with odd seq_len
  if (seq_len % 2 != 0){
    if(seq_len <= 32)
      block.x = 32;
    else if(seq_len > 32 && seq_len <= 64)
      block.x = 64;
    else if(seq_len > 64 && seq_len <= 128)
      block.x = 128;
    else if(seq_len > 128 && seq_len <= 256)
      block.x = 256;
    else if(seq_len > 256 && seq_len <= 512)
      block.x = 512;
    else
      block.x = 1024;

    if(batch_size * head_num <= 120)
    {
      grid.x = batch_size * head_num * seq_len;
      softmax_kernel_v2<T><<<grid, block, 0, stream>>>(qk_buf_, attr_mask, batch_size, head_num, seq_len, scaler);
    }
    else
    {
      grid.x = batch_size * head_num;
      softmax_kernel<T><<<grid, block, 0, stream>>>(qk_buf_, attr_mask, batch_size, head_num, seq_len, scaler);
    }
  }
  //deal with even seq_len
  else{
    grid.x = seq_len;
    if (batch_size * head_num > 360)
      grid.x = ceil(float(seq_len)/32.0f);
    grid.y = batch_size;
    grid.z = head_num;
    if (seq_len <= 32){
      block.x = 32;
      softmax_kernel_v3_LE32<T><<<grid, block, 0, stream>>>(qk_buf_, attr_mask, batch_size, head_num, seq_len, scaler);
    }
    else{
      if (sizeof(T) == 2){
        block.x = (seq_len/2 + 31)/32*32;
        softmax_kernel_v3<<<grid, block, 0, stream>>>(qk_buf_, attr_mask, batch_size, head_num, seq_len, scaler);
      }
      else{
        block.x = (seq_len + 31)/32*32;
        softmax_kernel_v3<T><<<grid, block, 0, stream>>>(qk_buf_, attr_mask, batch_size, head_num, seq_len, scaler);
      }
    }
    grid.x = grid.y = grid.z = 1;
  }
}

template void softmax_kernel_kernelLauncher<float>(
    float* qk_buf_, const float* attr_mask,
    const int batch_size, const int head_num, const int seq_len,
    const float scaler,
    const cudaStream_t stream);

template void softmax_kernel_kernelLauncher<__half>(
    __half* qk_buf_, const __half* attr_mask,
    const int batch_size, const int head_num, const int seq_len,
    const __half scaler,
    const cudaStream_t stream);

/// *********************************** fin ***********************************


/// ****************** transpose & rm padding for attention *******************
template<typename T>
__global__
void transpose_rm_padding(
    T* src, T* dst,
    const int batch_size, const int seq_len,
    const int head_num, const int size_per_head,
    const int* batch_idx, const int* word_idx)
{
  int head_id  = threadIdx.y;
  int tid      = threadIdx.x;
  int batch_id = batch_idx[blockIdx.x];
  int word_id  = word_idx[blockIdx.x];

  int src_offset = batch_id * head_num * seq_len * size_per_head +
                   head_id * seq_len * size_per_head +
                   word_id * size_per_head +
                   tid;
  int dst_offset = blockIdx.x * head_num * size_per_head +
                   head_id * size_per_head +
                   tid;

  T* src_ptr = (T*)src;
  T* dst_ptr = (T*)dst;
  dst_ptr[dst_offset] = src_ptr[src_offset];
}

template<>
__global__
void transpose_rm_padding(
    __half* src, __half* dst,
    const int batch_size, const int seq_len,
    const int head_num, const int size_per_head,
    const int* batch_idx, const int* word_idx)
{
  // if (threadIdx.y == (head_num - 1) && threadIdx.x >= size_per_head)
  //   return;
  int head_id  = threadIdx.y;
  int tid      = threadIdx.x;
  int batch_id = batch_idx[blockIdx.x];
  int word_id  = word_idx[blockIdx.x];

  int src_offset = batch_id * head_num * seq_len * size_per_head +
                   head_id * seq_len * size_per_head +
                   word_id * size_per_head +
                   tid;
  int dst_offset = blockIdx.x * head_num * size_per_head +
                   head_id * size_per_head +
                   tid;

  half2* src_ptr = (half2*)src;
  half2* dst_ptr = (half2*)dst;
  dst_ptr[dst_offset] = src_ptr[src_offset];
}

template <typename T>
void transpose_rm_padding_kernelLauncher(
    T* src, T* dst,
    const int valid_word_num,
    const int batch_size, const int seq_len,
    const int head_num, const int size_per_head,
    const int* batch_idx, const int* word_idx,
    const cudaStream_t stream)
{
  dim3 grid(valid_word_num);
  dim3 block(size_per_head, head_num);

  transpose_rm_padding<float><<<grid, block, 0, stream>>>(
    src, dst,
    batch_size, seq_len, head_num, size_per_head,
    batch_idx, word_idx);
}

template <>
void transpose_rm_padding_kernelLauncher<__half>(
    __half* src, __half* dst,
    const int valid_word_num,
    const int batch_size, const int seq_len,
    const int head_num, const int size_per_head,
    const int* batch_idx, const int* word_idx,
    const cudaStream_t stream)
{
  dim3 grid(valid_word_num);
  dim3 block(size_per_head / 2, head_num);

  transpose_rm_padding<__half><<<grid, block, 0, stream>>>(
    src, dst,
    batch_size, seq_len, head_num, size_per_head / 2,
    batch_idx, word_idx);
}

template void transpose_rm_padding_kernelLauncher<float>(
  float* src, float* dst,
  const int valid_word_num,
  const int batch_size, const int seq_len,
  const int head_num, const int size_per_head,
  const int* batch_idx, const int* word_idx,
  const cudaStream_t stream);

template void transpose_rm_padding_kernelLauncher<__half>(
    __half* src, __half* dst,
    const int valid_word_num,
    const int batch_size, const int seq_len,
    const int head_num, const int size_per_head,
    const int* batch_idx, const int* word_idx,
    const cudaStream_t stream);

/// *********************************** fin ***********************************

}//namespace cuda
}//namespace effectivetransformer
