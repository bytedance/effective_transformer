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

#include "cuda_kernels.h"
#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <cuda_fp16.h>

namespace effectivetransformer{

// gelu code from
// https://github.com/NVIDIA/DeepLearningExamples/blob/master/FasterTransformer/v1/fastertransformer/cuda/cuda_kernels.cu#L26-L45
template <typename T>
__inline__ __device__
T gelu(T x)
{
  float cdf = 0.5f *
    (1.0f + tanhf((0.7978845608028654f * (x + 0.044715f * x * x * x))));
  return x * cdf;
}

template <>
__inline__ __device__
half2 gelu(half2 val)
{
  half2 val_pow3 = __hmul2(val, __hmul2(val, val));
  float2 tmp_pow = __half22float2(val_pow3);
  float2 tmp =  __half22float2(val);

  tmp.x = 0.5f *
    (1.0f + tanhf((0.7978845608028654f * (tmp.x + 0.044715f * tmp_pow.x))));
  tmp.y = 0.5f *
    (1.0f + tanhf((0.7978845608028654f * (tmp.y + 0.044715f * tmp_pow.y))));
  return __hmul2(val, __float22half2_rn(tmp));
}

// reduce code from
// https://github.com/NVIDIA/DeepLearningExamples/blob/master/FasterTransformer/v1/fastertransformer/cuda/cuda_kernels.cu#L47-L73

#define FINAL_MASK 0xffffffff

template <typename T>
__inline__ __device__
T warpReduceSum(T val)
{
  for(int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}

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

  val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : (T)0.0f;
  val = warpReduceSum(val);
  return val;
}

/// ***************************** add_bias + gelu *****************************

template <typename T>
__global__
void add_bias_act(T* out, const T* bias, int m, int n)
{
  T val, reg_bias;

  int row_id = blockIdx.x;
  int ite = n / blockDim.x;
  int tid = threadIdx.x;

  for(int i = 0; i < ite; ++i)
  {
    reg_bias = __ldg(&bias[i * blockDim.x + tid]);
    row_id = blockIdx.x;

    while(row_id < m){
      val = out[tid + i * blockDim.x + row_id * n]+ reg_bias;
      out[tid + i * blockDim.x + row_id * n] = gelu<T>(val);
      row_id += gridDim.x;
    }
  }
}

template <>
__global__
void add_bias_act(__half* out, const __half* bias, int m, int n)
{
  half2 val, reg_bias;
  int row_id = blockIdx.x;
  int ite = n / blockDim.x / 2;
  int tid = threadIdx.x;

  half2* out_ptr = (half2*) out;
  const half2* bias_ptr = (half2*) bias;
  for(int i = 0; i < ite; ++i)
  {
    reg_bias = __ldg(&bias_ptr[i * blockDim.x + tid]);
    row_id = blockIdx.x;

    while(row_id < m){
      val = out_ptr[tid + i * blockDim.x + row_id * n / 2];
      val = __hadd2(val, reg_bias);
      out_ptr[tid + i * blockDim.x + row_id * n / 2] = gelu<half2>(val);
      row_id += gridDim.x;
    }
  }
}

template <typename T>
void add_bias_act_kernelLauncher(
  T* out, const T* bias, int m, int n, cudaStream_t stream)
{
  dim3 grid(max(m / 4, 1));
  dim3 block(n / 4);
  assert(block.x < 1024);
  add_bias_act<T><<<grid, block, 0, stream>>>(out, bias, m, n);
}

template void add_bias_act_kernelLauncher<__half>(
  __half* out, const __half* bias, int m, int n, cudaStream_t stream);

template void add_bias_act_kernelLauncher<float>(
  float* out, const float* bias, int m, int n, cudaStream_t stream);

/// *********************************** fin ***********************************


/// ************************** add_bias + layer_norm **************************

template <typename T>
__global__
void add_bias_input_layernorm(
  T* out, const T* input, const T* bias, const T* gamma,
  const T* beta, int m, int n)
{
  int tid = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float local_out = 0.0f;
  for(int i = tid; i < n; i += blockDim.x)
    local_out += (float)(out[blockIdx.x * n + i]
                    + input[blockIdx.x * n + i] + __ldg(&bias[i]));

  mean = blockReduceSum<float>(local_out);
  if(threadIdx.x == 0)
    s_mean = mean / n;
  __syncthreads();

  variance = blockReduceSum<float>((
      local_out - s_mean) * (local_out - s_mean));
  if(threadIdx.x == 0)
    s_variance = variance / n + 1e-6f;
  __syncthreads();

  for(int i = tid; i < n; i += blockDim.x)
    out[blockIdx.x * n + i] =
	    (T)(((local_out - s_mean) * rsqrtf(s_variance))
      * (float)(__ldg(&gamma[i])) + (float)(__ldg(&beta[i])));
}

template <>
__global__
void add_bias_input_layernorm(
  __half* out, const __half* input, const __half* bias,
  const __half* gamma, const __half* beta, int m, int n)
{
  int tid = threadIdx.x;
  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;
  float2 local_out_fp2;

  half2* out_ptr = (half2*)out;
  const half2* input_ptr = (const half2*)input;
  const half2* bias_ptr = (const half2*)bias;
  const half2* gamma_ptr = (const half2*)gamma;
  const half2* beta_ptr = (const half2*)beta;

  float local_out = 0.0f;
  int id = blockIdx.x * n / 2 + tid;
  local_out_fp2 = __half22float2(
    __hadd2(__hadd2(out_ptr[id], input_ptr[id]), __ldg(&bias_ptr[tid])));
  local_out += local_out_fp2.x;
  local_out += local_out_fp2.y;

  mean = blockReduceSum<float>(local_out);
  if(threadIdx.x == 0)
    s_mean = mean / n;
  __syncthreads();

  variance = (local_out_fp2.x - s_mean) * (local_out_fp2.x - s_mean);
  variance += (local_out_fp2.y - s_mean) * (local_out_fp2.y - s_mean);
  variance = blockReduceSum<float>(variance);
  if(threadIdx.x == 0)
    s_variance = rsqrtf(variance / n + 1e-6f);
  __syncthreads();

  float2 gamma_val = __half22float2(__ldg(&gamma_ptr[tid]));
  float2 beta_val = __half22float2(__ldg(&beta_ptr[tid]));
  local_out_fp2.x =
    (local_out_fp2.x - s_mean) * s_variance * gamma_val.x + beta_val.x;
  local_out_fp2.y =
    (local_out_fp2.y - s_mean) * s_variance * gamma_val.y + beta_val.y;
  out_ptr[id] = __float22half2_rn(local_out_fp2);
}

template <typename T>
__global__
void add_bias_input_layernorm_v2(
  T* out, const T* __restrict input, const T* __restrict bias,
  const T* __restrict gamma, const T* __restrict beta, int n)
{
  const int ite = 4;
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;
  float local_out[ite];

  float sum = 0.0f;
  #pragma unroll
  for(int i = 0; i < ite; i++)
  {
    int col_id = i * blockDim.x + tid;
    int id = bid * n + col_id;
    local_out[i] = (float)(out[id] + __ldg(&input[id]) + __ldg(&bias[col_id]));
    sum += local_out[i];
  }

  mean = blockReduceSum<float>(sum);
  if(tid == 0)
    s_mean = mean / n;
  __syncthreads();

  float var = 0.0f;
  #pragma unroll
  for(int i = 0; i < ite; i++)
  {
    float diff = local_out[i] - s_mean;
    var += diff * diff;
  }

  variance = blockReduceSum<float>(var);
  if(tid == 0)
    s_variance = rsqrtf(variance / n + 1e-6f);
  __syncthreads();

  #pragma unroll
  for(int i = 0; i < ite; i++)
  {
    int col_id = i * blockDim.x + tid;
    int id = bid * n + col_id;
    out[id] = (T)((local_out[i] - s_mean) * s_variance * (float)__ldg(&gamma[col_id]) + (float)__ldg(&beta[col_id]));
  }
}

template <>
__global__
void add_bias_input_layernorm_v2(
  half* out, const half* __restrict input, const half* __restrict bias,
  const half* __restrict gamma, const half* __restrict beta, int n)
{
  const int ite = 4;
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;
  half2 local_out_half2[ite];

  half2* out_ptr = (half2*)out;
  const half2* input_ptr = (const half2*)input;
  const half2* bias_ptr = (const half2*)bias;
  const half2* gamma_ptr = (const half2*)gamma;
  const half2* beta_ptr = (const half2*)beta;

  // float sum = 0.0f;
  half2 sum = __float2half2_rn(0.0f);
  #pragma unroll
  for(int i = 0; i < ite; i++)
  {
    int col_id = i * blockDim.x + tid;
    int id = bid * n / 2 + col_id;
    local_out_half2[i] = out_ptr[id] + __ldg(&input_ptr[id]) + __ldg(&bias_ptr[col_id]);
    sum += local_out_half2[i];
  }

  mean = blockReduceSum<float>((float)(sum.x + sum.y));
  if(threadIdx.x == 0)
    s_mean = mean / n;
  __syncthreads();

  float var = 0.0f;
  half2 s_mean_2 = __float2half2_rn(s_mean);
  #pragma unroll
  for(int i = 0; i < ite; i++)
  {
    local_out_half2[i] = local_out_half2[i] - s_mean_2;
    float v1 = (float)local_out_half2[i].x;
    float v2 = (float)local_out_half2[i].y;
    var += v1 * v1 + v2 * v2;
  }

  variance = blockReduceSum<float>(var);
  if(threadIdx.x == 0)
    s_variance = rsqrtf(variance / n + 1e-6f);
  __syncthreads();

  half2 s_var_2 = __float2half2_rn(s_variance);
  #pragma unroll
  for(int i = 0; i < ite; i++)
  {
    int col_id = i * blockDim.x + tid;
    int id = bid * n / 2 + col_id;
    out_ptr[id] = local_out_half2[i] * s_var_2 * __ldg(&gamma_ptr[col_id]) + __ldg(&beta_ptr[col_id]);
  }
}

template<typename T>
void add_bias_input_layernorm_kernelLauncher(
  T* out, const T* input, const T* bias,
  const T* gamma, const T* beta, int m, int n, cudaStream_t stream)
{
  assert(n < 1024);
  dim3 grid(m);
  dim3 block(n);
  if(n == 768 || n == 1024)
    add_bias_input_layernorm_v2<T><<<grid, n / 4, 0, stream>>>(out, input, bias, gamma, beta, n);
  else
    add_bias_input_layernorm<T><<<grid, block, 0, stream>>>(out, input, bias, gamma, beta, m, n);
}

template <>
void add_bias_input_layernorm_kernelLauncher(
  __half* out, const __half* input, const __half* bias,
  const __half* gamma, const __half* beta, int m, int n, cudaStream_t stream)
{
  assert(n / 2 < 1024);
  dim3 grid(m);
  dim3 block(n / 2);
  if(m >= 512 && (n == 768 || n == 1024))
    add_bias_input_layernorm_v2<half><<<grid, n / 8, 0, stream>>>(out, input, bias, gamma, beta, n);
  else
    add_bias_input_layernorm<half><<<grid, block, 0, stream>>>(out, input, bias, gamma, beta, m, n);
}

template void add_bias_input_layernorm_kernelLauncher<float>(
  float* out, const float* input,
  const float* bias, const float* gamma, const float* beta,
  int m, int n, cudaStream_t stream);

template void add_bias_input_layernorm_kernelLauncher<__half>(
  __half* out, const __half* input,
  const __half* bias, const __half* gamma, const __half* beta,
  int m, int n, cudaStream_t stream);

/// *********************************** fin ***********************************


/// *********************** compresse transformer input ***********************

template <typename T>
__global__
void compress_bert_input(
  const T* from_tensor, const int* mask, const int* prefix_sum,
  T* to_tensor, int* batch_idx, int* word_idx,
  int batch_size , int seq_len, int hidden_dim)
{
  int bid = blockIdx.y;  // batch
  int wid = blockIdx.x;  // word
  int tid = threadIdx.x; //

  /// 1. count pos for from tensor
  int mask_idx  = bid * seq_len + wid;

  if (mask[mask_idx] > 0.5) {
    int valid_idx = prefix_sum[mask_idx];

    /// 2. wirte batch id and word id for each word
    if (tid == 0) {
      batch_idx[valid_idx] = bid;
      word_idx[valid_idx]  = wid;
    }

    /// 3. copy src data
    float* src_ptr = (float*)from_tensor;
    float* dst_ptr = (float*)to_tensor;
    int src_idx = mask_idx  * hidden_dim + tid;
    int dst_idx = valid_idx * hidden_dim + tid;
    dst_ptr[dst_idx] = src_ptr[src_idx];
  }
}

template <>
__global__
void compress_bert_input(
    const __half* from_tensor, const int* mask, const int* prefix_sum,
    __half* to_tensor, int* batch_idx, int* word_idx,
    int batch_size , int seq_len, int hidden_dim)
{
  int bid = blockIdx.y;  // batch
  int wid = blockIdx.x;  // word
  int tid = threadIdx.x; //

  /// 1. count pos for from tensor
  int mask_idx  = bid * seq_len + wid;

  if (mask[mask_idx] > 0.5) {
    int valid_idx = prefix_sum[mask_idx];

    /// 2. wirte batch id and word id for each word
    if (tid == 0) {
      batch_idx[valid_idx] = bid;
      word_idx[valid_idx]  = wid;
    }

    /// 3. copy src data
    half2* src_ptr = (half2*)from_tensor;
    half2* dst_ptr = (half2*)to_tensor;
    int src_idx = mask_idx  * hidden_dim + tid;
    int dst_idx = valid_idx * hidden_dim + tid;
    dst_ptr[dst_idx] = src_ptr[src_idx];
  }
}

template<typename T>
void compressBertInput_kernelLauncher(
    const T* from_tensor, const int* mask, const int* prefix_sum,
    T* to_tensor, int* batch_idx, int* word_idx,
    int batch_size , int seq_len, int hidden_dim, cudaStream_t stream)
{
  /// TODO : fp32
  dim3 grid(seq_len, batch_size);
  dim3 block(hidden_dim);
  // dim3 block(1);
  assert(hidden_dim <= 1024);
  compress_bert_input<<<grid, block, 0, stream>>>(
    from_tensor, mask, prefix_sum,
    to_tensor, batch_idx, word_idx,
    batch_size , seq_len, hidden_dim);
  return;
}

template<>
void compressBertInput_kernelLauncher(
    const __half* from_tensor, const int* mask, const int* prefix_sum,
    __half* to_tensor, int* batch_idx, int* word_idx,
    int batch_size , int seq_len, int hidden_dim, cudaStream_t stream)
{
  dim3 grid(seq_len, batch_size);
  dim3 block(hidden_dim / 2);
  // dim3 block(1);
  assert(hidden_dim <= 1024 / 2);
  compress_bert_input<<<grid, block, 0, stream>>>(
    from_tensor, mask, prefix_sum,
    to_tensor, batch_idx, word_idx,
    batch_size , seq_len, hidden_dim / 2);
}

template void compressBertInput_kernelLauncher<float>(
  const float* from_tensor, const int* mask, const int* prefix_sum,
  float* to_tensor, int* batch_idx, int* word_idx,
  int batch_size , int seq_len, int hidden_dim, cudaStream_t stream);

template void compressBertInput_kernelLauncher<__half>(
  const __half* from_tensor, const int* mask, const int* prefix_sum,
  __half* to_tensor, int* batch_idx, int* word_idx,
  int batch_size , int seq_len, int hidden_dim, cudaStream_t stream);

/// *********************************** fin ***********************************

/// *********************** restore transformer output ************************
template<typename T>
__global__
void restore_bert_output(
    T* to_tensor,
    const T* from_tensor, const int*  batch_idx, const int* word_idx,
    int valid_word_num, int seq_len, int hidden_dim)
{
  int bid = batch_idx[blockIdx.x];
  int wid = word_idx[blockIdx.x];
  int tid = threadIdx.x;
  int vid = blockIdx.x;

  /// 3. copy src data
  float* src_ptr = (float*)from_tensor;
  float* dst_ptr = (float*)to_tensor;
  int src_idx = vid * hidden_dim + tid;
  int dst_idx = (bid * seq_len + wid) * hidden_dim + tid;
  dst_ptr[dst_idx] = src_ptr[src_idx];
}

template <>
__global__
void restore_bert_output(
    __half* to_tensor,
    const __half* from_tensor, const int*  batch_idx, const int* word_idx,
    int valid_word_num, int seq_len, int hidden_dim)
{
  int bid = batch_idx[blockIdx.x];
  int wid = word_idx[blockIdx.x];
  int tid = threadIdx.x;
  int vid = blockIdx.x;

  /// 3. copy src data
  half2* src_ptr = (half2*)from_tensor;
  half2* dst_ptr = (half2*)to_tensor;
  int src_idx = vid * hidden_dim + tid;
  int dst_idx = (bid * seq_len + wid) * hidden_dim + tid;
  dst_ptr[dst_idx] = src_ptr[src_idx];
}

template<typename T>
void restoreBertOutput_kernelLauncher(
    T* to_tensor,
    const T* from_tensor, const int* batch_idx, const int* word_idx,
    int valid_word_num, int seq_len, int hidden_dim, cudaStream_t stream)
{
  // TODO : fp32
  dim3 grid(valid_word_num);
  dim3 block(hidden_dim);
  assert(hidden_dim <= 1024);
  restore_bert_output<<<grid, block, 0, stream>>>(
    to_tensor,
    from_tensor, batch_idx, word_idx,
    valid_word_num, seq_len, hidden_dim);
}

template<>
void restoreBertOutput_kernelLauncher(
    __half* to_tensor,
    const __half* from_tensor, const int* batch_idx, const int* word_idx,
    int valid_word_num, int seq_len, int hidden_dim, cudaStream_t stream)
{
  dim3 grid(valid_word_num);
  dim3 block(hidden_dim / 2);
  assert(hidden_dim <= 1024 / 2);
  restore_bert_output<<<grid, block, 0, stream>>>(
    to_tensor,
    from_tensor, batch_idx, word_idx,
    valid_word_num, seq_len, hidden_dim / 2);
}

template void restoreBertOutput_kernelLauncher<float>(
  float* to_tensor,
  const float* from_tensor, const int*  batch_idx, const int* word_idx,
  int valid_word_num, int seq_len, int hidden_dim, cudaStream_t stream);

template void restoreBertOutput_kernelLauncher<__half>(
  __half* to_tensor,
  const __half* from_tensor, const int*  batch_idx, const int* word_idx,
  int valid_word_num, int seq_len, int hidden_dim, cudaStream_t stream);

/// *********************************** fin ***********************************

/// ***************************** exclusive scan ******************************
// The scan code is rewritten based on this repo :
// https://github.com/mattdean1/cuda/tree/master/parallel-scan
// I only rewritted device memory allocation part.

int THREADS_PER_BLOCK = 512;
int ELEMENTS_PER_BLOCK = THREADS_PER_BLOCK * 2;
#define SHARED_MEMORY_BANKS 32
#define LOG_MEM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_MEM_BANKS)

__global__ void prescan_large(int *output, const int *input, int n, int *sums)
{
	extern __shared__ int temp[];

	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * n;

	int ai = threadID;
	int bi = threadID + (n / 2);
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
	temp[ai + bankOffsetA] = input[blockOffset + ai];
	temp[bi + bankOffsetB] = input[blockOffset + bi];

	int offset = 1;
	for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
	{
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	__syncthreads();


	if (threadID == 0) {
		sums[blockID] = temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)];
		temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
	}

	for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	output[blockOffset + ai] = temp[ai + bankOffsetA];
	output[blockOffset + bi] = temp[bi + bankOffsetB];
}

__global__ void prescan_arbitrary(
  int *output, const int *input, int n, int powerOfTwo)
{
	extern __shared__ int temp[];// allocated on invocation
	int threadID = threadIdx.x;

	int ai = threadID;
	int bi = threadID + (n / 2);
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);


	if (threadID < n) {
		temp[ai + bankOffsetA] = input[ai];
		temp[bi + bankOffsetB] = input[bi];
	}
	else {
		temp[ai + bankOffsetA] = 0;
		temp[bi + bankOffsetB] = 0;
	}


	int offset = 1;
  // build sum in place up the tree
	for (int d = powerOfTwo >> 1; d > 0; d >>= 1)
	{
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	if (threadID == 0) {
    // clear the last element
		temp[powerOfTwo - 1 + CONFLICT_FREE_OFFSET(powerOfTwo - 1)] = 0;
	}

	for (int d = 1; d < powerOfTwo; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	if (threadID < n) {
		output[ai] = temp[ai + bankOffsetA];
		output[bi] = temp[bi + bankOffsetB];
	}
}

__global__ void add(int *output, int length, int *n) {
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * length;

	output[blockOffset + threadID] += n[blockID];
}

__global__ void add(int *output, int length, const  int *n1, const int *n2) {
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * length;

	output[blockOffset + threadID] += n1[blockID] + n2[blockID];
}

// from https://stackoverflow.com/a/12506181
int nextPowerOfTwo(int x) {
	int power = 1;
	while (power < x) {
		power *= 2;
	}
	return power;
}

void scanSmallDeviceArray(
  int *d_out, const int* d_in, const int length, const cudaStream_t stream);
void scanLargeDeviceArray(
  int *d_out, const int* d_in, const int length, int *d_buf,
  const cudaStream_t stream);
void scanLargeEvenDeviceArray(
  int *d_out, const int* d_in, const int length, int *d_buf,
  const cudaStream_t stream);

void scanLargeEvenDeviceArray(
  int *d_out, const int* d_in, const int length, int *d_buf,
  const cudaStream_t stream)
{
	const int blocks = length / ELEMENTS_PER_BLOCK;
	const int sharedMemArraySize = ELEMENTS_PER_BLOCK * sizeof(int);

  int *d_sums = d_buf;
  int *d_incr = d_buf + blocks;
	// cudaMalloc((void **)&d_sums, blocks * sizeof(int));
	// cudaMalloc((void **)&d_incr, blocks * sizeof(int));

	prescan_large<<<blocks, THREADS_PER_BLOCK, 2 * sharedMemArraySize, stream>>>(
    d_out, d_in, ELEMENTS_PER_BLOCK, d_sums);

	const int sumsArrThreadsNeeded = (blocks + 1) / 2;
	if (sumsArrThreadsNeeded > THREADS_PER_BLOCK) {
		// perform a large scan on the sums arr
		scanLargeDeviceArray(d_incr, d_sums, blocks, d_buf, stream);
	}
	else {
		// only need one block to scan sums arr so can use small scan
		scanSmallDeviceArray(d_incr, d_sums, blocks, stream);
	}

	add<<<blocks, ELEMENTS_PER_BLOCK, 0, stream>>>(
    d_out, ELEMENTS_PER_BLOCK, d_incr);
}

void scanSmallDeviceArray(
  int *d_out, const int* d_in, const int length, const cudaStream_t stream)
{
	int powerOfTwo = nextPowerOfTwo(length);
	prescan_arbitrary
    <<<1, (length + 1) / 2, 2 * powerOfTwo * sizeof(int), stream >>>(
      d_out, d_in, length, powerOfTwo);
}

///
void scanLargeDeviceArray(
    int *d_out, const int* d_in, const int length, int *d_buf,
    const cudaStream_t stream)
{
	int remainder = length % (ELEMENTS_PER_BLOCK);
	if (remainder == 0) {
		scanLargeEvenDeviceArray(d_out, d_in, length, d_buf, stream);
	}
	else {
		// perform a large scan on a compatible multiple of elements
		int lengthMultiple = length - remainder;
		scanLargeEvenDeviceArray(d_out, d_in, lengthMultiple, d_buf, stream);

		// scan the remaining elements and add the (inclusive)
    // last element of the large scan to this
		int *startOfOutputArray = &(d_out[lengthMultiple]);
		scanSmallDeviceArray(
      startOfOutputArray, &(d_in[lengthMultiple]), remainder, stream);

		add<<<1, remainder, 0, stream>>>(
      startOfOutputArray, remainder, &(d_in[lengthMultiple - 1]),
      &(d_out[lengthMultiple - 1]));
	}
}

void exclusiveScan_kernelLauncher(
  int* d_out, const int* d_in, const int length, const cudaStream_t stream)
{
	if (length > ELEMENTS_PER_BLOCK) {
		scanLargeDeviceArray(d_out, d_in, length, d_out + length, stream);
	}
	else {
		scanSmallDeviceArray(d_out, d_in, length, stream);
	}
}

/// *********************************** fin ***********************************

}//namespace effectivetransformer
