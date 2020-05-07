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

#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
namespace effectivetransformer{

template <typename T>
void add_bias_act_kernelLauncher(T* out, const T* bias, int m, int n, cudaStream_t stream);

template <typename T>
void add_bias_input_layernorm_kernelLauncher(T* out, const T* input_tensor, const T* bias, 
  const T* gamma, const T* beta, int m, int n, cudaStream_t stream);

void exclusiveScan_kernelLauncher(int* d_out, const int* d_in, const int length, const cudaStream_t stream);

template<typename T>
void compressBertInput_kernelLauncher(
    const T* from_tensor, const int* mask, const int* prefix_sum, 
    T* to_tensor, int* batch_idx, int* word_idx,
    int batch_size , int seq_len, int hidden_dim, cudaStream_t stream);

template<typename T>
void restoreBertOutput_kernelLauncher(
  T* to_tensor,
  const T* from_tensor, const int*  batch_idx, const int* word_idx, 
  int valid_word_num, int seq_len, int hidden_size, cudaStream_t stream);

}//namespace effectivetransformer
